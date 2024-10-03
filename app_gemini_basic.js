// This imports the new Gemini LLM
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

// This imports the mechanism that helps create the messages
// called `prompts` we send to the LLM
import { PromptTemplate } from "langchain/prompts";

// This imports the tool called `chains` that helps combine 
// the model and prompts so we can communicate with the LLM
import { LLMChain } from "langchain/chains";

// This helps connect to our .env file
import * as dotenv from "dotenv";
dotenv.config();



const template = `quien es mario alberto kempes`;



const promptTemplate = new PromptTemplate({
  template
});

// Above we created a template variable that contains our
// detailed instructions for the LLM, we also added a 
// variable {emojis} which would be replaced with the emojis
// passed in at runtime.
// We then create a prompt template from the template and
// input variable.

// We create our model and pass it our model name 
// which is `gemini-pro`. Another option is to pass
// `gemini-pro-vision` if we were also sending an image
// in our prompt
const geminiModel = new ChatGoogleGenerativeAI({
  modelName: "gemini-pro",
});

// We then use a chain to combine our LLM with our 
// prompt template
const llmChain = new LLMChain({
  llm: geminiModel,
  prompt: promptTemplate,
});

// We then call the chain to communicate with the LLM
// and pass in the emojis we want to be explained.
// Note that the property name `emojis` below must match the
// variable name in the template earlier created.
const result = await llmChain.call({
});

// Log result to the console
console.log(result.text);