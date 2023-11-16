from src.utils import load_data
from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


class Doc2VecWrapper:
  def __init__(self):
    self.tagged_documents = None
    self.model = None
    self.epochs = None
    self.original_data_mapping = None

  def init(self, vector_size, alpha, min_alpha, min_count, epochs):
    """
    Initializes the doc2vec model.

    vector_size: Dimensionality of the feature vectors.
    alpha: The initial learning rate.
    min_alpha: Learning rate will linearly drop to min_alpha as training progresses.
    min_count: Ignores all words with total frequency lower than this.
    epochs: Number of iterations (epochs) over the corpus.
    """
    self.model = Doc2Vec(vector_size=vector_size,
                         alpha=alpha,
                         min_alpha=min_alpha,
                         min_count=min_count)
    self.epochs = epochs

  def fit(self, tokenized_texts: list[list[str]]):
    """
    Fits the doc2vec model on the data.

    tokenized_texts: List of lists of tokens.
    """
    self._tag_data(tokenized_texts)
    self.model.build_vocab(self.tagged_documents)

    self.original_data_mapping = {
        f"DOC_{str(i)}": text for i, text in enumerate(tokenized_texts)}

  def _tag_data(self, tokenized_texts: list[list[str]]):
    """
    Tags the data for the doc2vec model.

    tokenized_texts: List of lists of tokens.
    """
    self.tagged_documents = [TaggedDocument(
        words=_d, tags=[f"DOC_{str(i)}"]) for i, _d in enumerate(tokenized_texts)]

  def train(self):
    """
    Trains a doc2vec model on the data.
    """

    for epoch in tqdm(range(self.epochs), desc='Training doc2vec', ascii=True, colour="#0077B5"):
      self.model.train(self.tagged_documents,
                       total_examples=self.model.corpus_count, epochs=1)
      # decrease the learning rate
      self.model.alpha -= 0.002
      # fix the learning rate, no decay
      self.model.min_alpha = self.model.alpha

  def infer(self, tokenized_text: list[str]):
    """
    Infers a vector for a given tokenized text.

    tokenized_text: List of tokens.

    returns: Vector representation of the text.
    """
    return self.model.infer_vector(tokenized_text)

  def most_similar(self, doc_tag, topn=10):
    """
    Finds the most similar documents to a given document.

    doc_tag: Tag of the document.
    topn: Number of similar documents to return.

    returns: List of tuples (tag, similarity).
    """
    return self.model.dv.most_similar(doc_tag, topn=topn)

  def most_similar_original_format(self, doc_tag, topn=10):
    """
    Finds the most similar documents to a given document.

    doc_tag: Tag of the document.
    topn: Number of similar documents to return.

    returns: List of tuples (tag, similarity).
    """
    return [(self.original_data_mapping[doc_tag], similarity) for doc_tag, similarity in self.most_similar(doc_tag, topn)]


if __name__ == '__main__':
  jobs = load_data(kind="processed")
  jobs_descriptions = jobs['description'].tolist()

  doc2vec = Doc2VecWrapper()
  doc2vec.init(vector_size=50, alpha=0.025,
               min_alpha=0.00025, min_count=1, epochs=100)
  doc2vec.fit(jobs_descriptions)
  doc2vec.train()

  print(doc2vec.infer(["you", "are", "a", "very", "good", "programmer"]))

  # Original format query
  print(doc2vec.original_data_mapping["DOC_50"])

  similar_docs = doc2vec.most_similar_original_format("DOC_50")

  for doc in similar_docs:
    print(doc)
    print(50 * "-")
