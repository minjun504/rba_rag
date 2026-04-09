| Experiment | Chunk Size | Overlap | Model        | MRR   | Hit Rate | Sim   |
|------------|------------|---------|--------------|-------|----------|-------|
| baseline   | 512        | 64      | MiniLM-L6-v2 | 0.312 | 0.750    | 0.513 |
| exp1       | 256        | 64      | MiniLM-L6-v2 | 0.463 | 0.875    | 0.538 |
| exp2       | 128        | 64      | MiniLM-L6-v2 | 0.198 | 0.500    | 0.585 |
| exp3       | 256        | 128     | MiniLM-L6-v2 | 0.338 | 0.500    | 0.553 |
| exp4       | 256        | 64      | all-mpnet-base-v2 | 0.525 | 0.750    | 0.670 |
| exp4       | 256        | 32      | all-mpnet-base-v2 | 0.344 | 0.750    | 0.678 |