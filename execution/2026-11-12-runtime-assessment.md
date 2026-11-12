# Runtime assessment — 12 November 2026

## Requested work and actual status

The user requested execution for one selected model, then installation of Vensim and saving progress to their GitHub. Random Walk was chosen for the first pilot. No new forward simulation, posterior inference, misspecification experiment, or training has run. There are no new scientific results to report yet.

## Evidence checked

- Device architecture is Apple Silicon (`arm64`), macOS 26.6.
- No Vensim application or engine was found in `/Applications` or on the command path during this assessment.
- The source-compatible public repository revision remains `d74a866f0476c07e93d786d297f90c0706281c33` according to the fetched GitHub tree.
- Published model assets are compiled Vensim `.vpmx` packages. The local `RandomWalk3.vpmx` exists; it is not a standalone Python simulator.
- `setup_functions/vensim_simulation_model.py` imports `venpy`, loads the packaged model, writes sensitivity-control files and invokes the Vensim engine. The notebook specifies Windows executable paths and `useDll=True`.
- Official Vensim documentation states the Vensim DLL is not available for Macintosh. Installing a Mac GUI application does not establish compatibility with this DLL wrapper.
- The historical Python environment was created for x86_64 on a prior Intel Mac. Its interpreter links and recorded adaptation scripts/results are not currently available as a complete runnable native pipeline. Historical reported success is not a result from this session.
- Earlier readable Random Walk source differs in release from the compiled checkpoint training model. Its existence does not establish baseline equivalence, and it has not been substituted silently.

## Installation progress

- A Mac Vensim PLE educational-download request was accepted by the official Vensim download center. Newsletter subscription was omitted.
- The response said download instructions were emailed to the user's university address; it did not return an installer URL.
- Installer download, signature validation, installation and launch verification are pending receipt of that URL. Vensim is NOT installed yet.
- No paid license or Windows virtual machine has been purchased or configured.

## Git progress

- Initialized local Git and created an explicit tracked-file allowlist for our own protocol, implementation and execution notes.
- Raw private conversations, unpublished manuscripts, author checkpoint binaries, installers and environments are excluded.
- GitHub CLI 2.104.0 was installed from the official release after SHA-256 verification. The user completed GitHub's device authorization flow.
- Created and verified a private remote: https://github.com/youssefsaiid21-a11y/npe-reliability-research .
- Initial progress commit `4a485ca9a6a2c75609b557acea12e8ebae29d065` was pushed and its remote main-branch hash matched the local commit.

## Required next steps

1. Obtain the official emailed Mac installer URL, download it and verify the signed application/package before installation.
2. Confirm the installed Mac edition's ability to open/run the authors' model packages. Do not assume its presence provides the Windows-only DLL interface or editable structural equations.
3. Establish a supported batch-simulation route and editable source for the selected perturbation. If the Mac route cannot do that, use an authorized compatible Windows environment or an explicitly approved and validated author-source port.
4. Restore a compatible native Python inference environment only for the chosen validated route, retaining hashes/version locks and two-thread limits.
5. Run source checks and the independent pilot before allocating main simulations. Keep failures and baseline mismatches visible.
6. Continue saving own research code/progress to the verified private GitHub repository; keep the tracked-file allowlist and exclusions in place.

## Public sources

- Authors' repository: https://github.com/ali-akhavan89/SD_Neural_Estimation_Benchmarking
- Vensim DLL documentation: https://www.vensim.com/documentation/dss_dll.html
- VenPy project and platform limitations: https://github.com/pbreach/venpy
- Official Mac/educational downloads: https://vensim.com/free-downloads/
- Installation instructions: https://www.vensim.com/documentation/20050.html

This file records an execution prerequisite assessment, not an experiment result.
