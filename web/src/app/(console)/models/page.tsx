import { consolePage } from "@/components/layout/console-page";
import { Table } from "@/components/ui/table";
import { utcTime } from "@/lib/format";

export const dynamic = "force-dynamic";

export default consolePage(
  "models",
  "Models",
  "Which models does each alias use, in what order, and what is loaded right now?",
  (body) => (
    <>
      <Table
        columns={["Alias", "Purpose", "Providers in fallback order"]}
        rows={body.aliases.map((alias) => ({
          key: alias.name,
          cells: [
            <code key="name">{alias.name}</code>,
            alias.purpose ?? "no purpose given",
            alias.candidates.map((c, index) => `${index + 1}. ${c.provider} ${c.model}`).join(", "),
          ],
        }))}
      />
      <h2>Provider files loaded</h2>
      <Table
        columns={["File", "Provider", "Kind", "State"]}
        rows={body.provider_files.map((file) => ({
          key: file.name,
          cells: [<code key="file">{file.name}</code>, file.provider, file.kind, file.enabled ? "enabled" : "disabled"],
        }))}
      />
      <p className="note">Policy last loaded {utcTime(body.loaded_at)}.</p>
    </>
  ),
);
