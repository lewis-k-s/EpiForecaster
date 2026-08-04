"""Public workflow targets; computational behavior lives in domain rules."""


if WORKFLOW_KIND == "train":
    rule train:
        input:
            str(TRAINING_EVIDENCE_TERMINAL)


    rule evaluate:
        input:
            str(EVALUATION_MANIFEST)


    rule train_evaluate:
        input:
            str(EVALUATION_MANIFEST)


elif WORKFLOW_KIND == "evaluate":
    rule evaluate:
        input:
            str(EVALUATION_MANIFEST)


elif WORKFLOW_KIND == "hpo":
    rule evaluate:
        input:
            str(EVALUATION_MANIFEST)


    rule hpo:
        input:
            str(CHECKPOINTS_CLEANED)


elif WORKFLOW_KIND == "ablation":
    rule ablation:
        input:
            str(ABLATION_MANIFEST)


elif WORKFLOW_KIND == "crossval":
    rule crossval:
        input:
            str(CROSSVAL_MANIFEST)


elif WORKFLOW_KIND == "preprocess":
    rule preprocess:
        input:
            str(PREPROCESSING_RECEIPT)
