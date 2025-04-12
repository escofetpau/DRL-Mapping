# DRL-Mapping

## Usage

Do a single run using the gates_per_slice stored in level1:

```bash
poetry run python main.py
```

Automatically run curriculum learning:
```bash
poetry run python main.py --curriculum
```

Display tensorboard plots:
```bash
poetry run tensorboard --logdir=runs
```