# Getting Started

## Train with a single GPU

- The default command is as follows:

  ```bash
  python train.py ${CONFIG_FILE}
  ```

## Train with multiple GPUs

- The default command is as follows:

  ```bash
  bash dist_train.sh ${CONFIG_FILE} ${GPU_NUM}
  ```

## Test with a single GPU

- The default command is as follows:

  ```bash
  python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
  ```

## Test with multiple GPUs

- The default command is as follows:

  ```bash
  bash dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}
  ```

## Finetune with a single GPU

- The default command is as follows:

  ```bash
  python finetune.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
  ```