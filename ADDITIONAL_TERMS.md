# MV-Forcing — Additional Terms

These Additional Terms supplement the **PolyForm Noncommercial License 1.0.0**
(the "Base License") in [`LICENSE`](./LICENSE). The Licensor offers MV-Forcing
(the "Software") under the Base License **only on the condition** that you also
agree to these Additional Terms. They narrow, and do not broaden, the Base
License. Where the Base License and these Additional Terms conflict, **these
Additional Terms control to the extent of the conflict.**

Capitalized terms not defined here have the meaning given in the Base License.

## 1. Noncommercial use only; commercial licensing

The Base License grants rights for noncommercial purposes only, and no
commercial license is granted by it or by this file. "Commercial purpose" means
use primarily intended for or directed toward commercial advantage or monetary
compensation, including incorporation into, or use to develop, a product or
service offered to third parties.

To use the Software for any commercial purpose, you must first obtain a separate
written commercial license from the Licensor.

> **Commercial-licensing contact:** galf251@gmail.com

Note separately that MV-Forcing requires [CUT3R](https://github.com/CUT3R/CUT3R)
at both training and inference time, and CUT3R is licensed under CC BY-NC-SA 4.0.
A commercial license from the Licensor therefore does **not**, by itself, make
commercial use of the full pipeline possible; you would also need rights to any
third-party component you rely on. See Section 3 and [`NOTICE.md`](./NOTICE.md).

## 2. Prohibited uses (military and surveillance)

Notwithstanding any permission granted by the Base License — including, for the
avoidance of doubt, its allowance of use by **government institutions** — you may
**not** use the Software, in whole or in part, nor any modification of or work
based on the Software, nor any model weights released by the Licensor for it, for
either of the following:

**(a) Military purposes.** Any use to research, design, develop, train, test,
deploy, operate, or support weapons, weapon systems, military operations, armed
forces, or any other military application.

**(b) Surveillance purposes.** Any use to conduct or materially facilitate
surveillance that identifies, tracks, profiles, or monitors individuals or
groups, including mass surveillance, biometric identification or tracking of
natural persons, predictive policing, or any monitoring carried out in violation
of internationally recognized human rights or applicable data-protection and
privacy law.

These prohibitions are conditions on all licenses granted. Any use described in
2(a) or 2(b) falls outside every license granted, and all of your licenses
terminate **immediately** upon such use (the 32-day cure period in the Base
License's "Violations" section does not apply to uses prohibited by this
Section 2).

## 3. Third-party components

The Base License and these Additional Terms cover **only** the Licensor's own
original MV-Forcing code and the model weights the Licensor releases for it.
Inherited, vendored and third-party components are not covered and remain
governed by their own licenses, including but not limited to:

- **Self-Forcing** — this repository is derived from
  https://github.com/guandeh17/Self-Forcing. Files inherited from it, whether
  unchanged or modified, remain under the **Apache License 2.0**; see
  [`licenses/LICENSE.Apache-2.0`](./licenses/LICENSE.Apache-2.0).
- **Wan2.1** — the `wan/` package, vendored through Self-Forcing from
  https://github.com/Wan-Video/Wan2.1, under the **Apache License 2.0**. The
  Wan2.1-T2V-1.3B weights are downloaded by the user under their own terms.
- **CUT3R** — https://github.com/CUT3R/CUT3R, under
  **CC BY-NC-SA 4.0**. Not redistributed here; the user supplies it. Its
  ShareAlike obligation applies to adaptations of CUT3R itself.
- **SynCamMaster / ReCamMaster** weights — not redistributed here; downloaded by
  the user from their respective releases under their own terms.
- External dependencies (e.g., PyTorch, diffusers, transformers) — under their
  own licenses.

See [`NOTICE.md`](./NOTICE.md) for details, including which files are inherited
unchanged and which were modified. Nothing in the Base License or these
Additional Terms grants any rights in those components beyond what their own
licenses provide.

## 4. No additional warranty

These Additional Terms add no warranty or liability of any kind. The Software is
provided "as is" as stated in the Base License's "No Liability" section.
