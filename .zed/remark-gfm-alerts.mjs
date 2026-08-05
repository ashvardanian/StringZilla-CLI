import { visit } from 'unist-util-visit'

// GitHub renders `> [!WARNING]` as a callout, but the marker is plain text to remark, and
// `mdast-util-to-markdown` escapes a leading `[` because it could open a link reference.
// Re-emitting the marker as a raw `html` node passes it through verbatim.
const MARKER = /^(\[!(?:NOTE|TIP|IMPORTANT|WARNING|CAUTION)\])(\n?)([\s\S]*)$/

export default function remarkGfmAlerts() {
  return (tree) => {
    visit(tree, 'blockquote', (node) => {
      const paragraph = node.children[0]
      if (paragraph?.type !== 'paragraph') return
      const first = paragraph.children[0]
      if (first?.type !== 'text') return
      const match = MARKER.exec(first.value)
      if (!match) return
      const [, marker, newline, rest] = match
      const replacement = [{ type: 'html', value: marker }]
      if (newline || rest) replacement.push({ type: 'text', value: newline + rest })
      paragraph.children.splice(0, 1, ...replacement)
    })
  }
}
