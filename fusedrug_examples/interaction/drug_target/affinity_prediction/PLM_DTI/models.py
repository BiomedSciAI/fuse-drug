import torch
import torch.nn as nn


class Cosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2)


class SquaredCosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2) ** 2


class Euclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0)


class SquaredEuclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0) ** 2


DISTANCE_METRICS = {
    "Cosine": Cosine,
    "SquaredCosine": SquaredCosine,
    "Euclidean": Euclidean,
    "SquaredEuclidean": SquaredEuclidean,
}

#######################
# Model Architectures #
#######################


class DeepCoembedding(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        model_dropout=0.2,
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation(),
            nn.Linear(latent_dimension, latent_dimension), latent_activation(),
            nn.Linear(latent_dimension, latent_dimension), latent_activation(),
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)
        nn.init.xavier_normal_(self.drug_projector[2].weight)
        nn.init.xavier_normal_(self.drug_projector[4].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation(),
            nn.Linear(latent_dimension, latent_dimension), latent_activation(),
            nn.Linear(latent_dimension, latent_dimension), latent_activation(),
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)
        nn.init.xavier_normal_(self.target_projector[2].weight)
        nn.init.xavier_normal_(self.target_projector[4].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


