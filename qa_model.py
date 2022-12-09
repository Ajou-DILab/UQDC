qar_tokenizer = AutoTokenizer.from_pretrained('yjernite/retribert-base-uncased', truncation = True)
qar_model = AutoModel.from_pretrained('yjernite/retribert-base-uncased').to("cuda:1")

_ = qar_model.eval()

qa_s2s_tokenizer = AutoTokenizer.from_pretrained('yjernite/bart_eli5', truncation = True)
qa_s2s_model = AutoModelForSeq2SeqLM.from_pretrained('yjernite/bart_eli5').to("cuda:1")

_ = qa_s2s_model.eval()

faiss_res = faiss.StandardGpuResources()
wiki40b_passage_reps = np.memmap(
            'wiki40b_passages_reps_32_l-8_h-768_b-512-512_1.dat',
            dtype='float32', mode='r',
            shape=(13000000, 128)
)

wiki40b_index_flat = faiss.IndexFlatIP(128)
wiki40b_gpu_index = faiss.index_cpu_to_gpu(faiss_res, 0, wiki40b_index_flat)
wiki40b_gpu_index.add(wiki40b_passage_reps)
print(wiki40b_gpu_index.ntotal)


glue = load_dataset('glue', 'qnli')


test_glue = glue['test']

df = pd.DataFrame({
    'question': test_glue['question'],
    'answers': test_glue['sentence']
})

nlp_rouge = nlp.load_metric('rouge')

from rouge_score import rouge_scorer
from rouge import Rouge

import random
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


scorer = rouge_scorer.RougeScorer(['rougeL'])

score = []
rouge = Rouge()
set_seed(0)
check = []
t_size = 1
from tqdm.auto import tqdm

questions = []
answers = []
reference = []
premise = None
hypothesis = None
for j in range(1):
    question = "How do you feel when an ugly guy tells you that you look like him?"
    print("qq", question)

    doc, res_list = query_qa_dense_index(
        question, qar_model, qar_tokenizer,
        wiki40b_snippets, wiki40b_gpu_index, device="cuda:1"
    )
    premise = [res['article_title'] + " " + res['passage_text'] for res in res_list]

    question_doc = "question: {} context: {}".format(question, doc)
    answer = qa_s2s_generate(
        question_doc, qa_s2s_model, qa_s2s_tokenizer,
        num_answers=1,
        min_len=64,
        max_len=256,
        num_beams= 8,
        max_input_length=1024,
        device="cuda:1",
        temp=1
    )[0]
    print(answer)
    questions += [question]
    answers += [answer]
    #reference += [eli5['test_eli5'][6]['answers']['text'][0]]
    #print("rr", reference)
    hypothesis = question + " " + answer

context_df = pd.DataFrame({
    'Article': ['---'] + [res['article_title'] for res in res_list],
    'Sections': ['---'] + [res['section_title'] if res['section_title'].strip() != '' else res['article_title']
                 for res in res_list],
    'Text': ['--- ' + question] + [res['passage_text'] for res in res_list],
})
df.style.set_properties(**{'text-align': 'left'})
