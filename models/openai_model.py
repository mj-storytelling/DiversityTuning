from openai import OpenAI
import time

class OpenAIModel:
  def __init__(self, model_name: str):
      self.model_name = model_name
      self.model = OpenAI()

  def generate(self, prompt: str, num_return_sequences: int = 1, accu=True):
    return_list = []
    for i in range(num_return_sequences):
      print(i)
      
      if len(return_list) > 0 and accu:
        
        cur_prompt = f"{prompt}\n\nTry to write a creative writing to be far from the given examples, in terms of the plot and style.\n\n\nExamples:"
        for r_idx, r in enumerate(return_list):
          cur_prompt += f"\n\n===========Example {r_idx+1}===========\n{r}"
        print("accu...", len(cur_prompt))
      else:
        cur_prompt = f"{prompt}"
      while True:
        try:
          if "o1" in self.model_name:
            completion = self.model.chat.completions.create(
              model=self.model_name,
              messages=[
                  {
                      "role": "developer",
                      "content": "You write a creative writing based on the user-given writing prompt."
                  },
                  {
                      "role": "user",
                      "content": cur_prompt
                  }
              ], 
              max_completion_tokens=2048,
              temperature=1.0,
              # top_p=0.95,
            )
          else:
            completion = self.model.chat.completions.create(
              model=self.model_name,
              messages=[
                  {
                      "role": "developer",
                      "content": "You write a creative writing based on the user-given writing prompt."
                  },
                  {
                      "role": "user",
                      "content": cur_prompt
                  }
              ], 
              max_tokens=2048,
              temperature=1.0,
              top_p=0.95,
            )
          # print(completion.choices[0].message.content)
          return_list.append(completion.choices[0].message.content)
          print("success!")
          time.sleep(0.1)
          
          break
        except Exception as e:
          print(e)
          time.sleep(2)


    return return_list
