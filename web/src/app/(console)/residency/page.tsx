import { consolePage } from "@/components/layout/console-page";
import { Table } from "@/components/ui/table";
import { NA } from "@/lib/format";

export const dynamic = "force-dynamic";

export default consolePage("residency", "Residency", (body) => (
  <>
    <h2>Requests by region</h2>
    <Table
      columns={["Tenant", "Region", "Provider", "Requests"]}
      rows={body.by_region.map((row) => ({
        key: `${row.tenant}/${row.region}/${row.provider}`,
        cells: [row.tenant, row.region, row.provider, row.requests],
      }))}
    />

    <p>Violations: {body.violations ?? NA}</p>

    <h2>Redactions</h2>
    <Table
      columns={["Tenant", "Kind", "Count"]}
      rows={body.redactions.map((row) => ({
        key: `${row.tenant}/${row.kind}`,
        cells: [row.tenant, row.kind, row.count],
      }))}
    />
  </>
));
