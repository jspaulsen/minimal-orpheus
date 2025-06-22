from dataclasses import dataclass


@dataclass
class HubGGUFModel:
    repo_id: str
    filename: str


# @dataclass
# class LocalModel:
#     model_path: str


Model = HubGGUFModel
