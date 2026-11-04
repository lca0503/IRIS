"""The MAEB tasks and instructions used for IRIS task generalization."""

TASKS_BY_TYPE = {
    "Classification": (
        "BeijingOpera",
        "BirdCLEF",
        "CommonLanguageAgeDetection",
        "GTZANGenre",
        "IEMOCAPGender",
        "MInDS14",
        "MridinghamTonic",
        "SIBFLEURS",
        "VoxPopuliLanguageID",
    ),
    "Clustering": ("VehicleSoundClustering", "VoxPopuliGenderClustering"),
    "PairClassification": (
        "NMSQAPairClassification",
        "VoxPopuliAccentPairClassification",
    ),
    "Reranking": ("GTZANAudioReranking",),
    "Any2AnyRetrieval": ("JamAltArtistA2ARetrieval",),
}
TASKS = tuple(task for group in TASKS_BY_TYPE.values() for task in group)
QUERY_DOCUMENT_TASKS = frozenset(
    TASKS_BY_TYPE["Reranking"] + TASKS_BY_TYPE["Any2AnyRetrieval"]
)
TASK_INSTRUCTIONS = {
    "BeijingOpera": "Find recordings with the same percussion instrument as the query.",
    "BirdCLEF": "Find recordings with the same bird species as the query.",
    "CommonLanguageAgeDetection": "Find recordings with the same speaker age group as the query.",
    "GTZANGenre": "Find recordings with the same music genre as the query.",
    "IEMOCAPGender": "Find recordings with the same gender as the query.",
    "MInDS14": "Find recordings with the same intent as the query.",
    "MridinghamTonic": "Find recordings with the same tonic as the query.",
    "SIBFLEURS": "Find recordings with the same topic as the query.",
    "VoxPopuliLanguageID": "Find recordings with the same spoken language as the query.",
    "VehicleSoundClustering": "Find recordings with the same environmental sound as the query.",
    "VoxPopuliGenderClustering": "Find recordings with the same gender as the query.",
    "NMSQAPairClassification": "Find recordings that contain the answer to the query.",
    "VoxPopuliAccentPairClassification": "Find recordings with the same accent as the query.",
    "GTZANAudioReranking": "Find recordings with the same music genre as the query.",
    "JamAltArtistA2ARetrieval": "Find recordings by the same artist as the query.",
}

# Display names keep the same column order as the paper's 15-task report.
TASK_COLUMNS = (
    "BeijingOpera",
    "BirdCLEF",
    "CommonLangAge",
    "GTZANGenre",
    "IEMOCAPGender",
    "MInDS14",
    "MridinghamTonic",
    "SIBFLEURS",
    "VoxPopuliLanguageID",
    "VehicleSound",
    "VoxPopuliGender",
    "NMSQA",
    "VoxPopuliAccent",
    "GTZAN",
    "JamAlt Artist A2A",
)
