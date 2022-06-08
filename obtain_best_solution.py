import gdown
import zipfile
from src.transformers import TransformersApproach
from src.utils import CustomTest

if __name__ == '__main__':

    # download best model and extract it
    url = "https://drive.google.com/uc?id=1uB52rwvPBPWjra3N1z9I80nY28Gj4P84"
    output = "zipped_bert_base_uncased_pos_split2.zip"
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('.')

    # build submission file
    test_path = 'dataset/test.tsv'

    t = TransformersApproach("checkpoint-15606")

    [test_formatted] = CustomTest(test_path).preprocess(t.tokenizer, mode="with_pos")

    t.compute_prediction(test_formatted, output_file='best_submission_score.csv')
