import { consolePage } from "@/components/layout/console-page";
import { ExternalLink } from "@/components/ui/external-link";
import { utcTime } from "@/lib/format";

export const dynamic = "force-dynamic";

export default consolePage("incidents", "Incidents", "What went wrong, and what did we learn?", (body) => (
  <ul>
    {body.incidents.map((incident) => (
      <li key={incident.id}>
        <h2>
          {incident.open && <strong>Open</strong>} {incident.severity} {incident.title}
        </h2>
        <p>{incident.summary}</p>
        <p>
          Started {utcTime(incident.started_at)}, resolved {utcTime(incident.resolved_at)}
        </p>
        <ExternalLink url={incident.postmortem_url} label="Postmortem" />
      </li>
    ))}
  </ul>
));
