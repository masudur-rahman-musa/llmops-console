# The console image. Two stages: dependencies and the Next build in one, the
# runtime carrying only the standalone server Next traces out of that build.
#
# Nothing secret is baked in. The password digest, the session secret and the
# admin key arrive as environment variables at run time, from `console-secrets`
# in the cluster.
#
# The base is pinned by digest, not by tag: `22-bookworm-slim` moves with every
# Node patch and Debian rebuild. The index behind this digest carries
# linux/arm64, which the CAX21 needs, and linux/amd64 for CI and laptops.
FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5 AS build

WORKDIR /app

# pnpm at the version `package.json` names, through the corepack Node ships.
RUN corepack enable

# The dependency layer, cached until the lockfile changes.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile

COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

FROM node:22-bookworm-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5 AS runtime

WORKDIR /app

ENV NODE_ENV=production \
    PORT=3001 \
    HOSTNAME=0.0.0.0 \
    NEXT_TELEMETRY_DISABLED=1

# The standalone output is `server.js` plus only the `node_modules` it traces as
# reached; `.next/static` is not part of it and is copied beside it. There is no
# `public/` directory today; the day one exists, it is copied here the same way.
COPY --from=build --chown=1000:1000 /app/.next/standalone ./
COPY --from=build --chown=1000:1000 /app/.next/static ./.next/static

# uid 1000 is the base image's own `node` user.
USER 1000

EXPOSE 3001

CMD ["node", "server.js"]
