"""Derive the knowledge graph by scanning the Iceberg artifacts + edges.

Nodes are people, files, AI sessions, and TOPIC nodes (salient terms extracted from
real commit messages and AI prompts/responses). Edges connect them:

  person --modified-->  file        (person authored a commit that modified the file)
  file   --co_edited--> file        (changed together in one commit)
  person --ran-->       ai_session  (the person ran the session)
  ai_session --ai_touched--> file   (the session referenced the real file)
  topic  --about-->     person/file/ai_session

Every node carries the real underlying artifact ids it derives from, so the UI can
link each node back to its source. No graph DB — this is computed in Python from the
two Iceberg tables (the only source of truth).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .ids import person_id
from .lakehouse import Lakehouse

logger = logging.getLogger(__name__)

# Terms that are never useful as topics (English + code/commit noise).
STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "have", "has", "are", "was",
    "were", "will", "would", "should", "could", "can", "not", "but", "you", "your",
    "all", "any", "out", "use", "used", "using", "add", "added", "adds", "fix", "fixed",
    "fixes", "update", "updated", "updates", "remove", "removed", "change", "changed",
    "changes", "new", "old", "now", "get", "set", "let", "via", "into", "onto", "than",
    "then", "when", "what", "which", "who", "how", "why", "where", "code", "file", "files",
    "function", "return", "import", "self", "none", "true", "false", "null", "value",
    "values", "test", "tests", "make", "made", "also", "more", "less", "some", "one", "two",
    "first", "last", "next", "main", "src", "app", "run", "running", "need", "needs", "like",
    "just", "only", "here", "there", "they", "them", "their", "its", "his", "her", "our",
    "able", "based", "type", "types", "data", "name", "names", "line", "lines", "page",
    "user", "users", "call", "calls", "called", "work", "works", "working", "around",
    "claude", "session", "tracevault",
    # Programming-language keywords / generic code noise (topics should be concepts).
    "interface", "export", "const", "var", "string", "number", "boolean", "void",
    "enum", "async", "await", "class", "public", "private", "static", "promise", "proto",
    "generated", "title", "label", "props", "component", "default", "module", "package",
    "json", "html", "css", "div", "span", "href", "http", "https", "www", "com", "org",
    "def", "elif", "args", "kwargs", "param", "params", "result", "results", "response",
    "request", "error", "errors", "exception", "throw", "catch", "try", "else", "while",
}
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+#-]{2,}")
MAX_TOPICS = 24
MAX_TOPIC_FILES = 20
MAX_NODE_ARTIFACTS = 60


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t.lower() for t in TOKEN_RE.findall(text)} - STOPWORDS


def _scope_to_repo(
    art_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]], repo: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Restrict artifacts/edges to a single project: its commits, files, the AI sessions
    whose cwd is that project, and those sessions' messages. Person ids are kept implicitly
    (they are not artifacts; an authored edge to a kept commit pulls the person in)."""
    keep: set[str] = set()
    sessions: set[str] = set()
    for r in art_rows:
        if r["kind"] in ("commit", "file") and r.get("repo") == repo:
            keep.add(r["id"])
        elif r["kind"] == "ai_session" and r.get("repo") == repo:
            keep.add(r["id"])
            sessions.add(r["id"])
    msg_session = {
        e["src_id"]: e["dst_id"] for e in edge_rows if e["relation"] == "message_of_session"
    }
    for r in art_rows:
        if r["kind"] == "ai_message" and msg_session.get(r["id"]) in sessions:
            keep.add(r["id"])

    def ok(nid: str) -> bool:
        return nid in keep or nid.startswith("person:")

    art_rows2 = [r for r in art_rows if r["id"] in keep]
    edge_rows2 = [e for e in edge_rows if ok(e["src_id"]) and ok(e["dst_id"])]
    return art_rows2, edge_rows2


def _person_display(actor: str | None) -> str:
    if not actor:
        return "unknown"
    name = actor.split("<", 1)[0].strip()
    return name or actor


@dataclass
class GraphData:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "edges": self.edges}


class GraphBuilder:
    """Builds the full derived graph once, then filters by focus / kind / size."""

    def __init__(self, lake: Lakehouse, repo: str | None = None) -> None:
        arts = lake.scan_artifacts()
        edges = lake.scan_edges()

        art_rows = arts.to_dicts() if arts.height else []
        edge_rows = edges.to_dicts() if edges.height else []
        if repo:
            art_rows, edge_rows = _scope_to_repo(art_rows, edge_rows, repo)
        self._scoped = repo is not None
        self._art_rows = art_rows
        self._edge_rows = edge_rows

        self.id_kind: dict[str, str] = {}
        self.id_title: dict[str, str] = {}
        self.person_label: dict[str, str] = {}
        self.commit_author: dict[str, str] = {}
        self.session_person: dict[str, str] = {}
        self.id_modality: dict[str, str] = {}
        self._art_tokens: dict[str, set[str]] = {}

        for row in self._art_rows:
            aid = row["id"]
            kind = row["kind"]
            self.id_kind[aid] = kind
            self.id_title[aid] = row.get("title") or aid
            if kind == "file" and row.get("extra"):
                try:
                    self.id_modality[aid] = json.loads(row["extra"]).get("modality", "")
                except (ValueError, TypeError):
                    pass
            actor = row.get("actor")
            if kind in ("commit", "ai_session") and actor:
                pid = person_id(actor)
                self.person_label.setdefault(pid, _person_display(actor))
                if kind == "commit":
                    self.commit_author[aid] = pid
                else:
                    self.session_person[aid] = pid
            # Topic corpus: human language only (commit messages + AI prompts/responses).
            # File code snippets are excluded — they swamp topics with language keywords.
            if kind in ("commit", "ai_message", "ai_session"):
                self._art_tokens[aid] = _tokens(
                    f"{row.get('title') or ''} {row.get('text') or ''}"
                )

        # Edges from Iceberg, split by relation and resolved by endpoint kind.
        self.commit_files: dict[str, list[str]] = defaultdict(list)
        self.message_session: dict[str, str] = {}
        self.co_edited: list[tuple[str, str, float]] = []
        self.ai_touched: list[tuple[str, str, float]] = []
        self.person_sessions: dict[str, set[str]] = defaultdict(set)
        self.person_commits: dict[str, set[str]] = defaultdict(set)

        for e in self._edge_rows:
            rel = e["relation"]
            src, dst = e["src_id"], e["dst_id"]
            w = float(e.get("weight") or 1.0)
            if rel == "modified":
                self.commit_files[src].append(dst)
            elif rel == "co_edited":
                self.co_edited.append((src, dst, w))
            elif rel == "ai_touched_file":
                self.ai_touched.append((src, dst, w))
            elif rel == "message_of_session":
                self.message_session[src] = dst
            elif rel == "authored":
                if self.id_kind.get(dst) == "ai_session":
                    self.person_sessions[src].add(dst)
                elif self.id_kind.get(dst) == "commit":
                    self.person_commits[src].add(dst)

        self._full = self._compute_full_graph()

    # --- full graph construction ---

    def _compute_full_graph(self) -> GraphData:
        person_file: Counter[tuple[str, str]] = Counter()
        for cid, pid in self.commit_author.items():
            for fid in self.commit_files.get(cid, []):
                person_file[(pid, fid)] += 1

        topics = self._extract_topics()

        # --- nodes ---
        nodes: dict[str, dict[str, Any]] = {}

        def add_node(nid: str, kind: str, label: str, artifacts: list[str], modality: str = "") -> None:
            if nid not in nodes:
                nodes[nid] = {
                    "data": {
                        "id": nid,
                        "label": label,
                        "kind": kind,
                        "modality": modality,
                        "artifacts": artifacts[:MAX_NODE_ARTIFACTS],
                        "artifact_count": len(artifacts),
                    }
                }

        # people (anyone who authored a commit or ran a session)
        people = set(self.person_commits) | set(self.person_sessions)
        for pid in people:
            related = sorted(self.person_commits.get(pid, set()) | self.person_sessions.get(pid, set()))
            add_node(pid, "person", self.person_label.get(pid, pid), related)
        # files
        for aid, kind in self.id_kind.items():
            if kind == "file":
                add_node(aid, "file", self.id_title.get(aid, aid), [aid], self.id_modality.get(aid, ""))
            elif kind == "ai_session":
                add_node(aid, "ai_session", self.id_title.get(aid, aid), [aid])
        # topics
        for term, info in topics.items():
            add_node(f"topic:{term}", "topic", term, sorted(info["artifacts"]))

        # --- edges ---
        edges: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add_edge(src: str, dst: str, relation: str, weight: float) -> None:
            if src not in nodes or dst not in nodes or src == dst:
                return
            eid = f"{relation}:{src}->{dst}"
            if eid in seen:
                return
            seen.add(eid)
            edges.append(
                {"data": {"id": eid, "source": src, "target": dst, "relation": relation, "weight": weight}}
            )

        for (pid, fid), count in person_file.items():
            add_edge(pid, fid, "modified", float(count))
        for a, b, w in self.co_edited:
            add_edge(a, b, "co_edited", w)
        for pid, sessions in self.person_sessions.items():
            for sid in sessions:
                add_edge(pid, sid, "ran", 1.0)
        for sid, fid, w in self.ai_touched:
            add_edge(sid, fid, "ai_touched", w)
        for term, info in topics.items():
            tnode = f"topic:{term}"
            for pid in info["people"]:
                add_edge(tnode, pid, "about", 1.0)
            for sid in info["sessions"]:
                add_edge(tnode, sid, "about", 1.0)
            for fid in list(info["files"])[:MAX_TOPIC_FILES]:
                add_edge(tnode, fid, "about", 1.0)

        return GraphData(nodes=list(nodes.values()), edges=edges)

    def _extract_topics(self) -> dict[str, dict]:
        """Salient terms across real commit/AI text -> topic membership (traceable)."""
        doc_freq: Counter[str] = Counter()
        for toks in self._art_tokens.values():
            for t in toks:
                doc_freq[t] += 1
        candidates = [t for t, n in doc_freq.most_common() if n >= 2]
        chosen = candidates[:MAX_TOPICS]

        topics: dict[str, dict] = {
            t: {"artifacts": set(), "people": set(), "files": set(), "sessions": set()}
            for t in chosen
        }
        chosen_set = set(chosen)
        for aid, toks in self._art_tokens.items():
            kind = self.id_kind.get(aid)
            for term in toks & chosen_set:
                info = topics[term]
                info["artifacts"].add(aid)
                if kind == "commit":
                    if aid in self.commit_author:
                        info["people"].add(self.commit_author[aid])
                    info["files"].update(self.commit_files.get(aid, []))
                elif kind == "file":
                    info["files"].add(aid)
                elif kind == "ai_session":
                    info["sessions"].add(aid)
                    if aid in self.session_person:
                        info["people"].add(self.session_person[aid])
                elif kind == "ai_message":
                    sid = self.message_session.get(aid)
                    if sid:
                        info["sessions"].add(sid)
                        if sid in self.session_person:
                            info["people"].add(self.session_person[sid])
        # Drop topics that ended up connected to nothing.
        return {t: info for t, info in topics.items() if info["people"] or info["files"] or info["sessions"]}

    # --- filtering ---

    def build(
        self,
        *,
        focus: str | None = None,
        kinds: list[str] | None = None,
        max_nodes: int = 220,
    ) -> GraphData:
        nodes_by_id = {n["data"]["id"]: n for n in self._full.nodes}
        edges = self._full.edges

        if focus and focus in nodes_by_id:
            keep = {focus}
            for e in edges:
                if e["data"]["source"] == focus:
                    keep.add(e["data"]["target"])
                elif e["data"]["target"] == focus:
                    keep.add(e["data"]["source"])
            kept_nodes = [nodes_by_id[i] for i in keep if i in nodes_by_id]
        else:
            degree: Counter[str] = Counter()
            for e in edges:
                degree[e["data"]["source"]] += 1
                degree[e["data"]["target"]] += 1

            def _is_file(nid: str) -> bool:
                return nodes_by_id.get(nid, {}).get("data", {}).get("kind") == "file"

            if self._scoped:
                # A single chosen project: show EVERY node (every file included).
                keep = {n["data"]["id"] for n in self._full.nodes}
            else:
                # Cross-project overview: people, topics, sessions, plus topic-linked files.
                topic_files = {
                    e["data"]["target"]
                    for e in edges
                    if e["data"]["relation"] == "about" and _is_file(e["data"]["target"])
                }
                keep = set()
                for n in self._full.nodes:
                    k = n["data"]["kind"]
                    nid = n["data"]["id"]
                    if k in ("person", "topic", "ai_session"):
                        keep.add(nid)
                    elif k == "file" and nid in topic_files:
                        keep.add(nid)
            if len(keep) > max_nodes:
                ranked = sorted(keep, key=lambda i: degree.get(i, 0), reverse=True)
                keep = set(ranked[:max_nodes])
            kept_nodes = [nodes_by_id[i] for i in keep]

        if kinds:
            kinds_set = set(kinds)
            kept_nodes = [n for n in kept_nodes if n["data"]["kind"] in kinds_set]

        kept_ids = {n["data"]["id"] for n in kept_nodes}
        kept_edges = [
            e for e in edges if e["data"]["source"] in kept_ids and e["data"]["target"] in kept_ids
        ]
        return GraphData(nodes=kept_nodes, edges=kept_edges)


def build_graph(
    lake: Lakehouse,
    *,
    repo: str | None = None,
    focus: str | None = None,
    kind: str | None = None,
    max_nodes: int = 220,
) -> dict[str, Any]:
    """Convenience: build and return the filtered graph as cytoscape-ready JSON."""
    kinds = [k.strip() for k in kind.split(",")] if kind else None
    return GraphBuilder(lake, repo=repo).build(focus=focus, kinds=kinds, max_nodes=max_nodes).as_dict()
