import { ExternalLink } from "@/components/ui/external-link";
import { SourceNotice } from "@/components/ui/source-notice";
import { Unavailable } from "@/components/ui/unavailable";
import { getConsolePage } from "@/lib/console-api";
import { shortSha, utcTime } from "@/lib/format";

export const dynamic = "force-dynamic";

function evalResult(passed: boolean | null): string {
  if (passed === null) return "no eval";
  return passed ? "passed" : "failed";
}

export default async function ReleasesPage() {
  const result = await getConsolePage("releases");
  if (!result.ok) {
    return <Unavailable title="Releases" reason={result.reason} />;
  }
  const body = result.body;
  return (
    <>
      <h1>Releases</h1>
      <SourceNotice sources={body.sources} />
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Commit</th>
              <th>Deployed</th>
              <th>Status</th>
              <th>Eval</th>
              <th>Pull request</th>
            </tr>
          </thead>
          <tbody>
            {body.releases.map((release) => (
              <tr key={`${release.git_sha}/${release.deployed_at}`}>
                <td>
                  <code>{shortSha(release.git_sha)}</code>
                </td>
                <td>{utcTime(release.deployed_at)}</td>
                <td>{release.status}</td>
                <td>{evalResult(release.eval_passed)}</td>
                <td>
                  <ExternalLink url={release.pr_url} label="PR" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
