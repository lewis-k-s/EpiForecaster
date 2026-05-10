#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${repo_root}/.claude/skills"
local_codex_skills_dir="${repo_root}/.codex/skills"
home_codex_skills_dir="${CODEX_HOME:-${HOME}/.codex}/skills"

if [[ ! -d "${source_dir}" ]]; then
  echo "No Claude skills directory found: ${source_dir}" >&2
  exit 1
fi

mkdir -p "${local_codex_skills_dir}"

# Remove home-level symlinks that point back into this repo. This keeps the
# project skills repo-scoped while preserving Codex system/runtime skills.
misplaced_link="${home_codex_skills_dir}/skills"
if [[ -L "${misplaced_link}" ]]; then
  link_target="$(readlink "${misplaced_link}")"
  if [[ "${link_target}" == "${source_dir}" ]]; then
    unlink "${misplaced_link}"
    echo "Removed misplaced symlink: ${misplaced_link}"
  fi
fi

linked_count=0
skipped_count=0
removed_home_count=0

for skill_dir in "${source_dir}"/*; do
  [[ -d "${skill_dir}" ]] || continue
  [[ -f "${skill_dir}/SKILL.md" ]] || continue

  skill_name="$(basename "${skill_dir}")"
  target="${local_codex_skills_dir}/${skill_name}"
  home_target="${home_codex_skills_dir}/${skill_name}"

  if [[ -e "${target}" && ! -L "${target}" ]]; then
    echo "Skipping ${skill_name}: ${target} exists and is not a symlink" >&2
    skipped_count=$((skipped_count + 1))
    continue
  fi

  ln -sfn "${skill_dir}" "${target}"
  echo "Linked ${skill_name} -> ${skill_dir}"
  linked_count=$((linked_count + 1))

  if [[ -L "${home_target}" ]]; then
    home_link_target="$(readlink "${home_target}")"
    if [[ "${home_link_target}" == "${repo_root}"/* ]]; then
      unlink "${home_target}"
      echo "Removed home-level project symlink: ${home_target}"
      removed_home_count=$((removed_home_count + 1))
    fi
  fi
done

echo "Done. Linked ${linked_count} local skill(s), skipped ${skipped_count}, removed ${removed_home_count} home symlink(s)."
