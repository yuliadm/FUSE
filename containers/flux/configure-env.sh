#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_path="${script_dir}/.env"

echo "FLUX.1 Fill [dev] requires each user to provide their own Hugging Face token."
echo "First accept the model terms:"
echo "  https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev"
echo "Then create a read token:"
echo "  https://huggingface.co/settings/tokens"
echo

if [[ -f "${env_path}" ]]; then
    read -r -p ".env already exists. Replace it? [y/N] " replace_env
    if [[ ! "${replace_env}" =~ ^[Yy]$ ]]; then
        echo "Kept the existing .env file."
        exit 0
    fi
fi

read -r -s -p "Paste your Hugging Face token (input hidden): " hf_token
echo

if [[ ! "${hf_token}" =~ ^hf_[A-Za-z0-9]+$ ]]; then
    echo >&2 "ERROR: The token does not look like a Hugging Face hf_ token."
    exit 2
fi

umask 077
{
    printf 'HF_TOKEN=%s\n' "${hf_token}"
    printf 'HOST_UID=%s\n' "$(id -u)"
    printf 'HOST_GID=%s\n' "$(id -g)"
} > "${env_path}"
chmod 600 "${env_path}"
unset hf_token

echo "Created ${env_path} with permissions 600."
echo "You can now run: docker compose up -d --force-recreate reference"

