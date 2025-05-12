from together import Together
import time

class TogetherModel:
  def __init__(self, model_name: str):
      self.model_name = model_name
      self.model = Together()

  def generate(self, prompt: str, num_return_sequences: int = 1):
    return_list = []
    for i in range(num_return_sequences):
      print(i)
      
      cur_prompt = f"{prompt}"
      while True:
        try:
          completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
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
            top_k=50,
            repetition_penalty=1.1,
          )
          # print(completion.choices[0].message.content)
          result = completion.choices[0].message.content
          # remove things between <think> and </think> without regex
          result = result[:result.index("<think>")] + result[result.index("</think>")+8:]




          return_list.append(result)
          print("success!", i, result)
          time.sleep(0.1)
          break
        except Exception as e:
          print(e)
          time.sleep(2)


    return return_list
