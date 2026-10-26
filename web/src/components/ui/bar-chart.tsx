const WIDTH = 600;
const HEIGHT = 200;
const PAD = 8;
const LEFT = 56;
const TOP = 12;
const BOTTOM = 24;
const LABEL_GAP = 48;
const BAR_SHARE = 0.7;
const SERIES_COLOURS = 4;

export type BarDay = { label: string; parts: ReadonlyArray<{ series: string; value: number }> };

/** Two decimals, so float noise never reaches an attribute. */
function round(value: number): number {
  return Math.round(value * 100) / 100;
}

/** `series-1` to `series-4` by the series' place in `series`, cycling past four. */
function colour(series: ReadonlyArray<string>, name: string): string {
  return `series-${(Math.max(series.indexOf(name), 0) % SERIES_COLOURS) + 1}`;
}

/**
 * A server-rendered stacked bar chart: no client script, no library, no inline style.
 * One bar per day, its parts stacked in `series` order and scaled together, so a
 * bar's parts add up to its total. A zero or negative part draws nothing. Colours
 * come from the `series-N` classes and keep each series' place, even when an earlier
 * series has nothing that day. No bars or only zeros renders `No data.`. A day label
 * closer than 48 px to the last one shown is skipped. Pinned by tests/bar-chart.test.tsx.
 */
export function BarChart({
  title,
  days,
  series,
  yFormat = String,
}: {
  title: string;
  days: ReadonlyArray<BarDay>;
  series: ReadonlyArray<string>;
  yFormat?: (value: number) => string;
}) {
  const totals = days.map((day) => day.parts.reduce((sum, part) => sum + Math.max(part.value, 0), 0));
  const max = Math.max(0, ...totals);
  if (days.length === 0 || !(max > 0)) {
    return (
      <figure>
        <figcaption>{title}</figcaption>
        <p className="note">No data.</p>
      </figure>
    );
  }
  const slot = (WIDTH - LEFT - PAD) / days.length;
  const barWidth = round(slot * BAR_SHARE);
  const plot = HEIGHT - TOP - BOTTOM;
  let lastLabel = Number.NEGATIVE_INFINITY;
  return (
    <figure>
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="chart" role="img" aria-label={title}>
        {[0, max].map((value) => (
          <text key={`y${value}`} x={LEFT - 6} y={HEIGHT - BOTTOM - (value / max) * plot + 4} textAnchor="end" className="chart-text">
            {yFormat(value)}
          </text>
        ))}
        <path d={`M${LEFT} ${TOP} V${HEIGHT - BOTTOM} H${WIDTH - PAD}`} fill="none" className="chart-axis" />
        {days.map((day, index) => {
          const x = round(LEFT + index * slot + (slot - barWidth) / 2);
          const centre = round(x + barWidth / 2);
          let base = HEIGHT - BOTTOM;
          const rects = series.flatMap((name) => {
            const value = day.parts
              .filter((part) => part.series === name)
              .reduce((sum, part) => sum + Math.max(part.value, 0), 0);
            if (value <= 0) return [];
            const height = round((value / max) * plot);
            base -= height;
            return [{ name, value, y: round(base), height }];
          });
          const showLabel = centre - lastLabel >= LABEL_GAP;
          if (showLabel) lastLabel = centre;
          return (
            <g key={`b${index}`} className="bar-stack">
              {rects.map((rect) => (
                <rect key={rect.name} x={x} y={rect.y} width={barWidth} height={rect.height} className={colour(series, rect.name)}>
                  <title>{`${day.label} ${rect.name}: ${yFormat(rect.value)}`}</title>
                </rect>
              ))}
              {showLabel && (
                <text x={centre} y={HEIGHT - 8} textAnchor="middle" className="chart-text">
                  {day.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <ul className="legend">
        {series.map((name) => (
          <li key={name} className="legend-item">
            <span className={`swatch ${colour(series, name)}`} aria-hidden="true" />
            {name}
          </li>
        ))}
      </ul>
    </figure>
  );
}
