from model.base import BaseModel


class StandardModel(BaseModel):
    def forward(self, inputs):
        convolved = self.convolutional(inputs)
        return self.dense(convolved)
