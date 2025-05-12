import anthropic
import time

# client = anthropic.Anthropic(
#     # defaults to os.environ.get("ANTHROPIC_API_KEY")
#     api_key="my_api_key",
# )
# message = client.messages.create(
#     model="claude-3-5-sonnet-20241022",
#     max_tokens=1024,
#     messages=[
#         {"role": "user", "content": "Hello, Claude"}
#     ]
# )
# print(message.content)
# import time

class AnthropicModel:
  def __init__(self, model_name: str):
      self.model_name = model_name
      self.model = anthropic.Anthropic()

  def generate(self, 
    prompt: str, 
    num_return_sequences: int = 1, 
    system="You write a creative writing based on the user-given writing prompt.",
    temperature = 1.0,
    top_p = 0.95,
    top_k = 50,
    max_tokens=2048,
    ):
    return_list = []
    for i in range(num_return_sequences):
      print(i)
      while True:
        try:
          if system == None:
            completion = self.model.messages.create(
              model=self.model_name,
              messages=[
                  {
                      "role": "user",
                      "content": prompt
                  }
              ], 
              max_tokens=max_tokens,
              temperature=temperature,
              top_p=top_p,
              top_k=top_k,
            )
          else:
            completion = self.model.messages.create(
              model=self.model_name,
              system=system,
              messages=[
                  {
                      "role": "user",
                      "content": prompt
                  }
              ], 
              max_tokens=max_tokens,
              temperature=temperature,
              top_p=top_p,
              top_k=top_k,
            )
          return_list.append(completion.content[0].text)
          print("success!")
          time.sleep(0.1)
          break
        except Exception as e:
          print(e)
          time.sleep(2)


    return return_list
