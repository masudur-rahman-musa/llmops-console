import { bool, listOf, nullable, shape, str } from "@/lib/shapes";

/** The models page body and its guard. It reads the live policy, so it carries no `sources`. */

export type ModelsBody = {
  loaded_at: string;
  aliases: Array<{ name: string; purpose: string | null; candidates: Array<{ provider: string; model: string }> }>;
  provider_files: Array<{ name: string; provider: string; kind: string; enabled: boolean }>;
};

const check = shape({
  loaded_at: str,
  aliases: listOf(shape({ name: str, purpose: nullable(str), candidates: listOf(shape({ provider: str, model: str })) })),
  provider_files: listOf(shape({ name: str, provider: str, kind: str, enabled: bool })),
});

export function modelsGuard(value: unknown): value is ModelsBody {
  return check(value);
}
