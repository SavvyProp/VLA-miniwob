# Third-party code and designs

- **MiniWoB++**, Farama Foundation and contributors, MIT License.
  Used as a pinned dependency; its package supplies native HTML/JavaScript assets,
  browser execution, environment code, and screenshots. The dependency retains
  its license: https://github.com/Farama-Foundation/miniwob-plusplus/blob/33c3b4ddef8c6eb67c57a29663d844b1eda7e614/LICENSE
- **CompWoB**, Google Research authors, Apache License 2.0.
  Selected composition definitions inform the task manifest. The legacy task
  files are not vendored or represented as unchanged implementations here:
  https://github.com/google-research/google-research/tree/83f1e7b59e762648c953de751d31f75720683622/compositional_rl/compwob
- **TinyVLA**, its authors, MIT License. Its source was audited for configuration
  and data/processor compatibility. The optional bridge imports the user's
  separately installed upstream implementation:
  https://github.com/liyaxuanliyaxuan/TinyVLA/tree/94f441827b45e4f76316ef6a0ae443736dc93a5d

No pretrained model weights are redistributed. Check their model-card terms
when downloading the pinned backbone. Complete revision metadata is in
`configs/upstream_sources.json`.
