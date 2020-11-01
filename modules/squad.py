import json

class SquadProcessor:
    def __init__(self):
        self.data = None

    def read_json(self, path):
        with open(path) as f:
            my_json = json.load(f)['data']
            self.data = my_json
    
    def to_dict(self):
        if self.data is None:
            print('Data is not loaded!')
            return
        
        qa_dict = { 'context': [], 'question': [], 'answer': [] }
        for text_obj in self.data:
            for paragraph in text_obj['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    if not qa['is_impossible']:
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            qa_dict['context'].append(context)
                            qa_dict['question'].append(question)
                            qa_dict['answer'].append(answer)
        
        return qa_dict



if __name__ == '__main__':
    processor = SquadProcessor()
    processor.read_json('../data/squad_train.json')