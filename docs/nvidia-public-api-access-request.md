# NVIDIA "Public API Endpoints" access request (draft)

Post this on the NVIDIA developer forums — e.g. as a reply in
["Request to enable 'Public API Endpoints' for my personal organization"](https://forums.developer.nvidia.com/t/request-to-enable-public-api-endpoints-for-my-personal-organization/377873).

---

## Title

Request to enable "Public API Endpoints" for my personal organization (image-generation models)

## Body

Hi NVIDIA team,

I'm building against the hosted NIM image-generation APIs from
[build.nvidia.com](https://build.nvidia.com) and believe my personal
organization needs the "Public API Endpoints" permission enabled.

**What works:**
- My `nvapi-...` key authenticates fine against the LLM/chat endpoints
  (`integrate.api.nvidia.com/v1` returns HTTP 200 immediately).
- The key was generated today under my personal NVIDIA developer account.

**What fails:**
- Every hosted image-generation endpoint hangs or errors:
  - `https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell`
    (POST `{prompt, width, height, steps, seed}`) — request hangs for several
    minutes, no response.
  - `https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3.5-large`
    — returns HTTP 404.
- The models are visible/selectable in the build.nvidia.com UI but calls from
  my key never complete.

I understand this symptom typically means the account/organization hasn't been
granted "Public API Endpoints" access for the image-gen models. Could you enable
that permission for my organization?

My account/org details:

- Developer account email: _(your email)_
- Organization: _(your org / Personal)_
- Key prefix: `nvapi-` (generated _(date)_)

Happy to provide the full key or anything else needed to verify. Thanks!

---
