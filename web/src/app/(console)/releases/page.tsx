import { consolePage } from "@/components/layout/console-page";
import { ExternalLink } from "@/components/ui/external-link";
import { Table } from "@/components/ui/table";
import { shortSha, utcTime } from "@/lib/format";

export const dynamic = "force-dynamic";

function evalResult(passed: boolean | null): string {
  if (passed === null) return "no eval";
  return passed ? "passed" : "failed";
}

export default consolePage("releases", "Releases", (body) => (
  <Table
    columns={["Commit", "Deployed", "Status", "Eval", "Pull request"]}
    rows={body.releases.map((release) => ({
      key: `${release.git_sha}/${release.deployed_at}`,
      cells: [
        <code key="sha">{shortSha(release.git_sha)}</code>,
        utcTime(release.deployed_at),
        release.status,
        evalResult(release.eval_passed),
        <ExternalLink key="pr" url={release.pr_url} label="PR" />,
      ],
    }))}
  />
));
