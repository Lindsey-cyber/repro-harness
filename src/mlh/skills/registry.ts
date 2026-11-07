// Every skill the harness can plan. Model-backed skills are planned only when the human enabled models.
import { BASE_SKILLS, type Skill } from "./index.ts";
import { VERIFY_SKILLS } from "./verify.ts";

export const SKILLS: Skill[] = [...BASE_SKILLS, ...VERIFY_SKILLS];

export function skill(id: string): Skill {
  const s = SKILLS.find((x) => x.id === id);
  if (!s) throw new Error(`unknown skill ${id}`);
  return s;
}
