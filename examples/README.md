# Minimal row replacement

```bash
python examples/position_replacement.py
```

The example illustrates role-resolved state replacement (paper Eq. 2). It
constructs two synthetic sequence-by-feature matrices and replaces the
recipient's scene rows with the donor's rows. Every other row stays unchanged.
It checks the position partition with the same partition helper used by the
experiments. No images, labels, network access, checkpoint, or GPU are needed.

The toy positions and matrix values are illustrative, not experimental data.
An actual experiment obtains positions from the pinned processor and prompt
layout, captures states at a specified decoder cut, and runs the patched state
through the remaining decoder. See [the method](../docs/method.md) and
[experiment entrypoints](../docs/source_layout.md) for those steps.
