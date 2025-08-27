# GCP--Gen-AI--Notes

Generative AI is a type of artificial intelligence that helps users create new content and ideas, including text, images, music, and even code. Yet, the way you apply gen AI can come in different forms. It's a technology that can be integrated into different applications, not an application itself.

Vertex AI is Google Cloud's unified machine learning (ML) platform. It empowers you to build, train, and deploy ML models and AI applications. For multiple modalities (text, code, images, speech), Vertex AI gives you access to Google's large generative AI models through Model Garden. You can tune Google's LLMs to meet your needs, and then deploy them for use in your AI-powered applications.

Vertex AI Search is a Google Cloud service that allows developers to easily integrate advanced search capabilities into their applications. It's a fully managed, scalable service that can handle large volumes of multimodal data and complex search queries.

Q:
generative AI- A type of artificial intelligence that can create new content, including images, text and music.
 Gemini is a generative AI model (or family of models) developed by Google
multimodal gen AI application-Using gen AI to analyze customer sentiment in video testimonials and survey data.


Foundation models are large AI models trained on enormous data sets. Traditionally, AI models were trained for a single purpose using specific data, like filtering spam emails from your inbox. Foundation models are different. They're trained on massive amounts of diverse data (text, images, and code), enabling them to adapt to many different tasks. This flexibility is what makes them so groundbreaking. LLMs or large language models are probably the most famous type of foundation model. They are a specialized type of foundation model that focuses specifically on language.
key features of Foundation models:
-Foundation models are trained on a wide variety of data, allowing them to learn general patterns and relationships that can be applied to different tasks.(trained on diverse data)
-With foundation models, one AI model can support a wide range of use cases.(flexible)
-Foundation models can be specialized for particular domains or use cases through additional, targeted training.(adaptable)
ex:
Gemini:Trained on a massive dataset of text, images, code, audio, video, and more. This multimodal training allows Gemini to perform tasks across various domains, including language understanding, image generation, code generation, and more.
Imegen:Trained primarily on a massive dataset of images and text descriptions. This enables Imagen to generate high-quality images from text descriptions, perform image editing tasks, and understand the content of images.
chirps:Trained on a large dataset of audio in various languages. Chirp is designed for speech recognition and can be used for tasks like voice assistants, transcription, and translation. Find information at the right time.
Q:LLMs are a specialized type of foundation model.,All LLMs are foundation models.

Foundation models, like other AI models, take in inputs, which are called prompts, to produce outputs.While all AI models have inputs or prompts, foundation models are usually much more flexible in what you can prompt. For example, some AI models have strict requirements on the type of inputs they can handle. They may only handle numbers, emails, or images.Foundation models, and especially multimodal foundation models, can take in a much broader set of inputs.Prompting is one of the most important skills in maximizing the value of gen AI models and tools.
Q:
1.What is the primary difference between foundation models and traditional AI models?
Foundation models are trained on massive amounts of diverse data for various tasks, while traditional models are trained on specific data for a single task.
2.defines a foundation model?Large AI models trained on a vast quantity of data, capable of adapting to a variety of tasks
3.What is the purpose of a prompt in the context of foundation models? To provide input to the model and trigger an output.
4.How do foundation models and prompt engineering work together to create value in generative AI? Foundation models offer a vast knowledge base, and prompt engineering guides the model to use this knowledge in responses.

2001: ML for spell check : Google begins using machine learning to help with spell check at scale in Google Search.
2006: Google Translate: Google launches Google Translate using machine learning to automatically translate languages.
2015: Google Search: Google integrates RankBrain into Google Search, using machine learning to analyze search queries and deliver more relevant results to users.
2016: TPUs: Google announces the Tensor Processing Unit, custom data center silicon built specifically for machine learning.
2018: Smart Compose in Gmail: Google announces Smart Compose, a new feature in Gmail that uses AI to help users more quickly reply to their email. Smart Compose builds on Smart Reply, another AI feature.
2023: Bard: Google releases Bard, an early experiment that lets people collaborate with generative AI.
2023: Gemini: Google introduces Gemini, a multimodal AI model capable of understanding and integrating various information types. The following year, Google expands the Gemini ecosystem with Gemini 1.5, bringing it to more products and launching Gemini Advanced, which provides access to Google's most capable AI models.

In 2008, Google launched Google Cloud, making these same cutting-edge technologies available to businesses everywhere. This allows individuals and businesses to benefit from Google's advancements in fields like AI, without having to start from scratch.

2012: Deep learning in speech recognition
A new era of AI begins when Google researchers improve speech recognition with deep neural networks, which is a new machine learning architecture loosely modeled after the neural structures of the brain.
2014: DeepMind acquisition: Google acquires one of the leading AI research labs in the world, DeepMind.
2015: Tensorflow: Google introduces a new scalable open source machine learning framework used in speech recognition with the release of TensorFlow.
2017: Transformer: Google releases “Attention is All You Need,” a research paper that introduces the transformer, a novel neural network architecture particularly well suited for language understanding, among many other things.
2018: AI Principles: Google publishes a set of guidelines the company follows when developing and using AI. For more information, see Google's AI Principles.
2018: BERT: Google introduces a new technique for natural language processing pre-training called BERT (Bidirectional Encoder Representations from Transformers), helping Search better understand users’ queries.
2024: Gemma: Google announces Gemma, a family of lightweight state-of-the-art open models built from the same research and technology used to create the Gemini models.

Google is an AI-first company, so you don’t have to be

Implementing a multi-directional strategy



Successful gen AI implementation requires a combination of top-down and bottom-up approaches. For both approaches, you want to be strategizing on:
Strategic focus
Exploration
Responsible AI
Resourcing
Impact
Continuous improvement

A creative matrix is great because it can help your team think through different aspects of what gen AI can do. For this creative matrix workflow, you would get a team together and create two axes. The axes are a bit flexible, but we can give you some ideas:
On one axis, you might write the different options for gen AI solutions. This can be organized by gen AI product, modality, or another dimension. On the other axis, you would write out things that are specific to your business. This can be business personas, workflows, or strategic goals.
Then, have the team write sticky notes (either virtually or in person) with different ideas of how the two axes can intersect and place them in the right spot.  This should get you to think creatively about how your organization can leverage gen AI to truly help your business priorities. 
For example, the overlap between the drive innovation section and Vertex AI Search could be that we create a search tool where customers can upload images to find similar faux floral arrangements. Or if we want to enhance customer experience with Vertex AI Conversation, we can build an AI-powered chatbot to provide instant customer support and answer questions about faux floral care, arrangement, and ordering.
Also, to increase efficiency, we can use Vertex AI studio to fine-tune a model to analyze customer data and predict future demand for different faux floral products.

Augmentation versus automation
Use gen AI to enhance or augment your strategic thinking for:
Critical thinking and problem solving: Gen AI can provide data and insights, but humans are still needed to interpret those insights and make informed decisions.
Creativity and innovation: Gen AI can assist in generating ideas and exploring possibilities, but human ingenuity is still essential for pushing boundaries and developing truly innovative solutions.
Relationship building and collaboration: Gen AI can facilitate communication and information sharing, but strong interpersonal skills are still crucial for building trust, fostering collaboration, and navigating complex human dynamics.
Strategic planning and vision: Gen AI can help with forecasting and trend analysis, but human leadership is essential for setting a long-term vision, defining goals, and charting a course for the future.
Use gen AI to automate tasks that are:
Repetitive and rule-based: data entry, information retrieval, content formatting, and basic code generation.
Time-consuming and resource-intensive: research, data analysis, content summarization, and initial draft creation.

Even for task automation, humans-in-the-loop are a necessary component for the gen AI implementation and continuous improvement. Use people for:
Data selection and preparation: ensuring that gen AI models are trained on high-quality, relevant data that is representative of the intended use cases.
Prompt design and refinement: crafting prompts that elicit accurate and useful responses from gen AI models.
Output evaluation and refinement: reviewing and editing gen AI-generated content to ensure accuracy, relevance, and alignment with brand guidelines.
Continuous monitoring and feedback: providing feedback on gen AI performance and identifying areas for improvement.

Q:
What is a key advantage of using Google Cloud's gen AI ecosystem for businesses?
It allows businesses to leverage Google's AI advancements without starting from scratch.
What is the recommended approach for businesses to effectively implement gen AI?
Combine a top-down strategic vision with bottom-up input from teams
Which of the following is a benefit of using Google Cloud for gen AI development?
It provides comprehensive services, including scalable infrastructure, enterprise-grade governance and security.
Why is it important for mid-level managers and individual contributors to be involved in gen AI adoption?Their proximity to workflows allows them to identify impactful gen AI solutions.


Artificial intelligence, or AI. AI is all about building or creating machines that normally require us humans to use our brains.Like learning, problem-solving, and decision-making.
machine learning, or ML for short. ML is all about using data to train machines to perform specific tasks.
The math itself is a bit beyond the scope of this learning path, but…essentially, we call the math the model. When you plug an input into the math or model, you receive an output similar to any mathematical equation.The model is often a complex structure with many parameters and algorithms that process input data to generate outputs. More like a system of equations.
 gen AI, short for generative AI, is a subset of machine learning that focuses on creating new content, like images, text, or music. 
Artificial intelligence (AI): The broad field of building machines that can perform tasks requiring human intelligence.
Machine learning (ML): A subset of AI where machines learn from data.
Generative AI: An application of AI that creates new content.

Data quality:
Accuracy-  If the data is inaccurate, the model will learn incorrect patterns and make faulty predictions. Imagine teaching a child about animals using a book with mislabeled pictures—they'd learn the wrong things. The same applies to AI.
Completeness-  Completeness refers to the size of a dataset as well as representation within the dataset. The size of the dataset is important because the model needs enough to make an accurate prediction. If a meteorologist tries to predict weather only based on the data of the past day, it will be a much worse prediction than if it used a much larger sample size.
Representative-  Data needs to be representative and inclusive, otherwise it can lead to skewed samples and biased outcomes. If a dataset about customer preferences is missing information about a certain demographic, the model might make inaccurate or biased generalizations about that group.
Consistency -Inconsistent data formats or labeling can confuse the model and hinder its ability to learn effectively. Imagine trying to assemble a puzzle where some pieces are labeled with numbers and others with letters—it would be a mess.
Relevance- The data must be relevant to the task the AI is designed to perform. For example, data about traffic patterns in London is unlikely to be helpful for predicting crop yields in Kansas.

Data accessibility: The ability of AI systems to effectively utilize this data is directly tied to data accessibility. Data accessibility ensures that the necessary data is readily available, usable, and of high quality, allowing for comprehensive model training and reducing potential biases. Without accessible data, even the most sophisticated algorithms are limited in their ability to learn and provide accurate predictions.

Data types: Data comes in various forms, just like information itself. We can broadly categorize this data into two main types: structured and unstructured.
The data they store for customer orders can include information like: 
Customer ID, Customer name, Purchase date, Order cost, Delivery address, Product image, Feedback (on a 1-5 star scale)
Structured data
Imagine your contact list on your phone. It has names, phone numbers, and maybe addresses, all organized in a list. That's structured data! It's easy to search and find the information you need.This type of data is often stored in something called a relational database, which is like a super-organized digital filing cabinet with information neatly arranged in tables. Other examples of structured data include things like online shopping orders or bank statements.
For the cleaning supply company’s database, this would include:
Customer ID, Customer name, Feedback (on a 1-5 star scale),Purchase date, Order cost

Unstructured data- 
Unstructured data lacks a predefined structure. It is messy and complex by nature. It cannot be easily organized into rows and columns, so more sophisticated analysis techniques are required. Examples of unstructured data include things like text documents (PDFs, emails, social media posts), images (photographs, digital artwork, medical scans), audio (speech recordings, music files), and video (movies, YouTube videos, smartphone videos).
For the cleaning supply company’s database, this would include: Feedback (free-form text), Product image, Email content

Machine learning approaches and data requirements
The quality, accessibility, and form of data are fundamental, but how that data is used for ML depends on the specific learning method. Machine learning has three primary learning approaches: supervised, unsupervised, and reinforcement learning, each with its own data requirements. 
Supervised models rely on labeled data, unsupervised models work with unlabeled data, and reinforcement learning learns through interaction and feedback.

Labeled data has tags, such as a name, type, or number. These tags, whether applied manually or by automated systems, assign meaning to the data.
For instance, an image dataset for training a cat-detection model would label each picture as either a cat or dog. Similarly, a set of customer reviews might be labeled as positive, negative, or neutral. These labels enable algorithms to learn relationships and make accurate predictions.
Unlabeled data is simply data that is not tagged or labeled in any way. It's raw, unprocessed information without inherent meaning. 
Examples of unlabeled data include a collection of unorganized photos, a stream of audio recordings, or website traffic logs without user categorization. In these cases, algorithms must independently discover patterns and structures within the data, as there are no pre-existing labels to guide the learning process.

Supervised machine learning trains models on labeled data, where each input is paired with its correct output, allowing the model to learn the relationship between them. The model's goal is to identify patterns and relationships within this labeled data, enabling it to accurately predict outputs for new, unseen inputs.
Predicting housing prices is a common example of supervised learning. A model is trained on a dataset where each house has labeled data, such as its size, number of bedrooms, location, and the corresponding sale price. This labeled data allows the algorithm to learn the relationship between the features of a house and its price. Once trained, the model can then predict the price of a new house based on its features.
Unsupervised ML models deal with raw, unlabeled data to find natural groupings. Instead of learning from labeled data, it dives headfirst into a sea of unlabeled data. 
For example, an unsupervised learning algorithm could analyze customer purchase history from your company's database. It might uncover hidden segments of customers with similar buying habits, even though you never explicitly labeled those segments beforehand. This can be incredibly valuable for targeted marketing or product recommendations.
Think of it as exploratory analysis. Unsupervised learning helps you understand the underlying structure of your data and uncover insights that you might not have even known to look for.

Reinforcement learning is all about learning through interaction and feedback. Imagine a robot learning to navigate a maze. It starts with no knowledge of the maze's layout. As it explores and interacts with the maze, it collects data—bumping into walls (negative feedback) or finding shortcuts (positive feedback). Through this process of trial and error, the algorithm learns which actions lead to the best outcomes.
It's like training a pet. You reward good behavior and discourage bad behavior. And over time, the pet learns to perform the desired actions. Similarly, in reinforcement learning, the algorithm learns to maximize rewards and minimize penalties by interacting with its environment.This type of learning is particularly useful in situations where you can't provide explicit instructions or labeled data. For example, you could use reinforcement learning to train a self-driving car to navigate complex traffic situations or to optimize the performance of a robot in a manufacturing plant.

Predictive maintenance with Vertex AI (supervised learning): By training a model on sensor data from machines like temperature, pressure, and vibration, Vertex AI can predict when a machine is likely to fail, enabling proactive maintenance and reducing downtime.
Anomaly Detection with BigQuery ML (unsupervised learning):  BigQuery ML can analyze historical transaction data (amount, location, time, etc.) to identify patterns and flag unusual transactions that deviate significantly from the norm. This helps prevent fraud and minimize financial losses.
Product recommendations with Vertex AI (reinforcement learning): Vertex AI can train a reinforcement learning model to recommend products to users based on their browsing history, purchase behavior, and other factors. The model learns to maximize user engagement and sales by continuously refining its recommendations.

Data tools and management for ML workloads:
Gather your data: 
Data gathering, also called data ingestion, involves collecting raw data from various sources. To effectively train and test your model, determine the data you need based on the outcome you want to achieve. Google Cloud supports data ingestion through several tools. 
Pub/Sub handles real-time streaming data processing, regardless of the structure of the data.
Cloud Storage is well-suited for storing unstructured data.
Cloud SQL and Cloud Spanner are used to manage structured data.
Prepare your data: 
Data preparation is the process of cleaning and transforming raw data into a usable format for analysis or model training. This involves formatting and labeling data properly. 
Google Cloud offers tools like BigQuery for data analysis and Data Catalog for data governance. These tools help prepare data for ML models. 
With BigQuery, you can filter data, correct its inconsistencies, and handle missing values. 
With Data Catalog, you can find relevant data for your ML projects. This tool provides a centralized repository to easily discover datasets in your organization.
Train your model: 
The process of creating your ML model using data is called model training. Google Cloud's Vertex AI platform provides a managed environment for training ML models. 
With Vertex AI, you can set parameters and build your model, using prebuilt containers for popular machine learning frameworks, custom training jobs, and tools for model evaluation. Vertex AI also provides access to powerful computing resources to make the model training process faster.
Deploy and predict:
Model deployment is the process of making a trained model available for use. 
Vertex AI simplifies this by providing tools to put the model into action for generating predictions. This includes scaling the deployment, which means adjusting the resources allocated to the model to handle varying amounts of usage.
Manage your model:
Model management is the process of managing and maintaining your models over time. Google Cloud offers tools for managing the entire lifecycle of ML models. This includes the following:
Versioning: Keep track of different versions of the model.
Performance tracking: Review the model metrics to check the model's performance.
Drift monitoring: Watch for changes in the model's accuracy over time.
Data management: Use Vertex AI Feature Store to manage the data features the model uses.
Storage: Use Vertex AI Model Garden to store and organize the models in one place.
Automate: Use Vertex AI Pipelines to automate your machine learning tasks.

Q:
How does consistency impact AI model training? Inconsistent formats and labeling can confuse the model and hinder learning.
What is the primary purpose of the data ingestion and preparation stage in the ML workflow?  Collecting, cleaning, and transforming raw data.
What is a "model" in the context of machine learning?  A complex mathematical structure that processes inputs to generate outputs
Which of the following is an example of unstructured data? A collection of customer reviews in the form of free-text paragraphs.
Arrange the ML lifecycle steps in the right order. Data ingestion and preparation > Model training > Model deployment > Model management
What is the primary way that agents learn in reinforcement learning? By interacting with their environment and receiving feedback.


Machine learning 
A broad field that encompasses many different techniques, one of which is deep learning (DL). 
Deep learning 
A powerful subset of machine learning, distinguished by its use of artificial neural networks. These networks enable the processing of highly complex patterns and the generation of sophisticated predictions.
Neural networks can leverage both labeled and unlabeled data, a strategy known as semi-supervised learning. They train on a blend of a small amount of labeled data and a large amount of unlabeled data. That way, they learn foundational concepts and generalize effectively to novel examples.
Generative AI uses the power of deep learning to create new content spanning text, images, audio, and beyond. Deep learning techniques, particularly those centered on neural networks, are the engine behind these generative models.
Foundation models use deep learning. They are trained on massive datasets that allow them to learn complex patterns and perform a variety of tasks across different domains. They are incredibly powerful machine learning models trained on a massive scale, often using vast amounts of unlabeled data. This training allows them to develop a broad understanding of the world, capturing intricate patterns and relationships within the data they consume.

Large language models (LLMs)​​​
One particularly exciting type of foundation model is the LLM. These models are specifically designed to understand and generate human language. 
They can translate languages, write different kinds of creative content, and answer your questions in an informative way, even if they are open ended, challenging, or strange. This is likely the most common foundation model you've encountered, such as in popular generative AI chatbots like Gemini. They also help power many search engines you use today.
Diffusion models
Diffusion models are another type of foundational model. They excel in generating high-quality images, audio, and even video by iteratively refining noise (or unstructured/random data and patterns) into structured data.

Factors when choosing a model for your use case:
Modality:
When selecting a generative AI model, it's crucial to consider the modality of your input and output. Modality refers to the type of data the model can process and generate, such as text, images, video, or audio. If your application focuses on a single data type, like generating text-based articles or creating audio files, you'll want to choose a model optimized for that specific modality. For applications that require handling multiple data types, such as generating image captions (processing images and producing text) or creating video with accompanying audio, you'll need a multimodal model. These models can understand and synthesize information across different modalities.
context window:
The context window refers to the amount of information a model can consider at one time when generating a response. A larger context window allows the model to "remember" more of the conversation or document, leading to more coherent and relevant outputs, especially for longer texts or complex tasks. However, larger context windows often come with increased computational costs. You need to balance the need for context with the practical limitations of your resources.
security:
Security is paramount, especially when dealing with sensitive data. Consider the model's security features, including data encryption, access controls, and vulnerability management. Ensure the model complies with relevant security standards and regulations for your industry.
Availability & Reliability:
The availability and reliability of the model are crucial for production applications. Choose a model that is consistently available and performs reliably under load. Consider factors like uptime guarantees, redundancy, and disaster recovery mechanisms.
cost:
Generative AI models can vary significantly in cost. Consider the pricing model, which might be based on usage, compute time, or other factors. Evaluate the cost-effectiveness of the model in relation to your budget and the expected value of your application. This is where selecting the right model for the right task is important. Be sure to match the model to the task; bigger isn't always better, and multi-modal capabilities aren't always necessary.
Performance:
The performance of the model, including its accuracy, speed, and efficiency, is a critical factor. Evaluate the model's performance on relevant benchmarks and datasets. Consider the trade-offs between performance and cost.
fine-tuning and customization:
Some models can be fine-tuned or customized for specific tasks or domains. If you have a specialized use case, consider models that offer fine-tuning capabilities. This often involves training the model further on a specific dataset related to your use case.
Ease of integrating:
The ease of integrating the model into your existing systems and workflows is an important consideration. Look for models that offer well-documented APIs and SDKs.

With Vertex AI you can access models developed by Google including Gemini, Gemma, Imagen, and Veo. You can also access proprietary third-party models, and openly available models.
Gemini:
Gemini, a multimodal model, can understand and operate across diverse data formats, such as text, images, audio, and video. Gemini's multimodal design supports applications that require complex multimodal understanding, advanced conversational AI, content creation, and nuanced question answering.
Gemma:
A family of lightweight, open models is built upon the research and technology behind Gemini. They offer developers a user-friendly and customizable solution for local deployments and specialized AI applications.
Imagen:
A powerful text-to-image diffusion model, it excels at generating high-quality images from textual descriptions. This makes it invaluable for creative design, ecommerce visualization, and content creation.
Veo:
A model capable of generating video content. It can produce videos based on textual descriptions or still images. Its functionality allows for the creation of moving images for applications such as film production, advertising, and online content.
Q-
Gemini is a multimodal model that can process and generate various data types, including text, images, audio, and video, while Gemma is a family of lightweight, open models suitable for local deployments and specialized AI applications.
Imagen is a text-to-image diffusion model that generates images from text, while Veo is a model capable of generating video content from text or images.

Foundation model limitations
Foundation models, while groundbreaking, aren't without limitations. Recognizing these limitations is essential for the responsible and effective utilization of these powerful tools.
Date dependency:
The performance of foundation models is heavily data-dependent. They require large datasets, and any biases or incompleteness in that data will inevitably seep into their outputs. It's like asking a student to write an essay on a book they haven't read. If the data or questions are inaccurate or biased, the AI's performance will suffer.
Knowledge cutoff:
Knowledge cutoff is the last date that an AI model was trained on new information. Models with older knowledge cutoffs may not know about recent events or discoveries. This can lead to incorrect or outdated answers, since AI models don't automatically update with the latest happenings around the world. 
For example, if an AI tool's last training date was in 2022, it wouldn't be able to provide information about events or information that happened after 2022. 
Bias:
An LLM learns from large amounts of data, which may contain biases. You can think of bias as an unbalanced dataset in LLMs. Due to their statistical learning nature, they can sometimes amplify existing biases present in the data. Even subtle biases in the training data can be magnified in the model's outputs.
Fairness:
Even with perfectly balanced data, defining what constitutes fairness in an LLM's output is a complex task. Fairness can be interpreted in various ways. Fairness assessments for generative AI models, while valuable, have inherent limitations. These evaluations typically focus on specific categories of bias, potentially overlooking other forms of prejudice. Consequently, these benchmarks do not provide a complete picture of all potential risks associated with the models' outputs, highlighting the ongoing challenge of achieving truly equitable AI.
Hallucinations:
Foundation models can sometimes experience hallucinations, which means they produce outputs that aren't accurate or based on real information. Because foundation models can't verify information against external sources, they may generate factually incorrect or nonsensical responses. These cause significant concern in accuracy-critical applications. The responses might sound convincing, but they are completely wrong. We will cover this in more detail below.
Edge cases:
Rare and atypical scenarios can expose a model's weaknesses, leading to errors, misinterpretations, and unexpected results. 

Techniques to overcome limitations
So, how do we address these challenges?
Fortunately, several techniques can significantly improve foundation model performance. Let's explore some of the key approaches.
Grounding
Generative AI models are amazing at creating content, but sometimes they hallucinate. Grounding is the process of connecting the AI's output to verifiable sources of information—like giving AI a reality check. By providing the model with access to specific data sources, we tether its output to real-world information, reducing the risk of invented content. 
Grounding is essential for building trustworthy and reliable AI applications. By connecting your models to verifiable data, you ensure accuracy and build confidence. It offers several key benefits, including reducing hallucinations, which prevents the AI from generating false or fictional information. Grounding also anchors responses, ensuring the AI's answers are rooted in your provided data sources. Furthermore, it builds trust by enhancing the trustworthiness of the AI's output by providing citations and confidence scores, allowing you to verify the information.

Retrieval-augmented generation (RAG)
There are many different options on how you can ground in data. For example, you can ground in enterprise data or you can ground using Google Search. One common grounding method to do this is with retrieval-augmented generation, or RAG.
RAG is a grounding method that uses search to find relevant information from a knowledge base and provides that information to the LLM, giving it necessary context.
The first step is retrieval. When you ask an AI a question, RAG uses a search engine to find relevant information. This search engine uses an index that understands the semantic meaning of the text, not just keywords. This means it finds information based on meaning, ensuring higher relevance. 
The retrieved information is then added to the prompt given to the AI. This is the augmentation phase. 
The AI then uses this augmented prompt, along with its existing knowledge, to generate a response. This is referred to as the generation phase.

Prompt engineering
Prompting offers the most rapid and straightforward approach to supplying supplementary background information to models. This involves crafting precise prompts to guide the model towards desired outputs. It refines results by understanding the factors that influence a model's responses. However, prompting is limited by the model's existing knowledge; it can't conjure information it hasn't learned.

Fine-tuning
When prompt engineering doesn't deliver the desired outcomes, fine-tuning can enhance your model's performance. Pre-trained models are powerful, but they're designed for general purposes. Tuning helps them excel in specific areas. This process is particularly useful for specific tasks or when you need to enforce specific output formats, especially if you have examples of the desired output.
Tuning involves further training a pre-trained or foundation model on a new dataset specific to your task. This process adjusts the model's parameters, making it more specialized for your needs. Google Cloud Vertex AI provides tooling to facilitate tuning.
Here are some examples of how tuning can be used:
Fine-tuning a language model to generate creative content in a specific style.
Fine-tuning a code generation model to generate code in a particular programming language.
Fine-tuning a translation model to translate between specific languages or domains.

Beyond these techniques, we must remember the invaluable role of humans in the loop (HITL). Machine learning models are powerful, but they sometimes need a human touch.
Content moderation:
HITL ensures accurate and contextually appropriate moderation of user-generated content, filtering out harmful or inappropriate material that algorithms alone might miss.
Sensitive applications:
In fields like healthcare or finance, HITL provides oversight for critical decisions, ensuring accuracy and mitigating risks associated with automated systems.
High-risk decision making:
When ML models inform decisions with significant consequences, such as medical diagnoses or criminal justice assessments, HITL acts as a safeguard, providing a layer of human review and accountability.
Pre-generation review:
Before deploying ML-generated content or decisions, human experts can review and validate the outputs, catching potential errors or biases before they impact users.
Post-generation review:
After ML outputs are deployed, continuous human review and feedback help identify areas for improvement, enabling models to adapt to evolving contexts and user needs.

Q; purpose of grounding in generative AI? To enhance the accuracy and reliability of AI-generated content by connecting it to verifiable sources.
Which techniques can be used to overcome the limitations of foundation model performance? Grounding, prompt engineering, fine-tuning, and humans in the loop (HITL).
Why is fine-tuning and customization an important factor when choosing a model? It allows the model to be adapted for specific tasks or domains.
Which type of foundation model would be most suitable for generating photorealistic images from textual descriptions? Diffusion model
What is the primary purpose of grounding in generative AI? To connect the AI's output to verifiable sources of information.
What is the primary role of humans in the loop (HITL) in machine learning? To integrate human expertise into the ML process, especially for tasks requiring judgment or context.
Which type of foundation model is specifically designed to understand and generate human language? Large language model (LLM)

What does Secure AI mean?
Secure AI is about preventing intentional harm being done to your applications. This is about protecting AI systems from malicious attacks and misuse. For all applications, including AI, you need to ensure security throughout the full lifecycle from development through deployment. This includes considering the data, infrastructure, and how and where applications are deployed.
Applying the Secure AI Framework (SAIF)​
Google has developed the Secure AI Framework, or SAIF, to establish security standards for building and deploying AI systems responsibly. This comprehensive approach to AI/ML model risk management addresses the key concerns of security professionals in the rapidly evolving landscape of AI. Following the security practices outlined in SAIF can help your organization find and stop threats, automatically strengthen its defenses, and manage the unique risks of each AI system. The framework is designed to integrate with your company’s existing security, ensuring that AI models are secure by default.
Platforms such as Google Cloud can facilitate secure development. In addition to the Secure AI Framework, Google Cloud offers a range of tools to ensure applications remain secure throughout their lifecycle. Google Cloud has built security into its core through a secure-by-design infrastructure, encompassing its global network and hardware, as well as robust encryption in transit and at rest. It provides customers with detailed control over access and usage of cloud resources through Identity and Access Management (IAM).
AI offers significant benefits, but it also introduces security risks like data poisoning, model theft, and prompt injection. To address these challenges, a secure foundation for AI applications is essential. Google Cloud's SAIF framework, combined with security tools, can help as you build and maintain secure AI systems.
Responsible AI:
Responsible AI means ensuring your AI applications avoid intentional and unintentional harm. It’s important to consider ethical development and positive outcomes for the software you develop. The same holds true, and perhaps even more so, for AI applications.The foundation of responsible AI is security. Secure applications protect both your company and your users. Think of it like building a house: if the foundation is weak, the entire structure is compromised, no matter how beautiful the design. Just as a strong foundation ensures a stable and safe house, robust security forms the essential foundation for building truly responsible AI.
Transparency is key: Transparency is paramount for responsible applications. Users need to understand how their information is being used and how the AI system works. This transparency should extend to all aspects of the AI's operation, including data handling, decision-making processes, and potential biases.
Privacy in the age of AI: Protecting privacy often involves anonymizing or pseudonymizing data, ensuring individuals can't be easily identified. AI models can sometimes inadvertently leak sensitive information from their training data, so it's crucial to implement safeguards to prevent this.
Data quality, bias, and fairness: 
Ethical AI requires high quality data:
Machine learning and generative AI applications are fundamentally based on data. Therefore, responsible AI requires high-quality data. Inaccurate or incomplete data can lead to biased outcomes. Remember, technology often reflects existing societal biases. Without careful consideration, AI can amplify these biases, leading to unfair and discriminatory results. Beyond data quality, it's crucial to consider the responsible use of the data itself. Was it collected consensually? Could it perpetuate harmful biases?
Understanding and mitigating bias​:
AI systems are not independent of the world they are built in. They can inherit and amplify existing societal biases. This can lead to unfair outcomes, such as a resume-screening tool that favors a certain demographic of candidates due to historical biases in hiring data. It's like training a dog with biased commands; the dog will learn and replicate those biases. To counter this, fairness must be a core principle in AI development.
Accountability and explainability:-
Fairness requires accountability​:
Fairness requires accountability. We need to know who is responsible for the AI's actions and understand how it makes decisions. This is where explainability comes in. Explainable AI makes the decision-making processes of AI models transparent and understandable. This is crucial for building trust, debugging errors, and uncovering hidden biases. Think of it like a judge explaining their verdict; without a clear explanation, it's hard to trust the decision. Tools like Google Cloud’s Vertex Explainable AI can help understand model outputs and identify potential biases. Understanding how your application uses and interprets the AI's output is crucial for ensuring responsible use.

Legal implications-
Beyond considerations like fairness and bias, AI development is increasingly governed by legal frameworks. Key areas include data privacy, non-discrimination, intellectual property, and product liability. 
Laws mandate responsible data handling, bias mitigation, and transparency in algorithmic decision-making.
Organizations must also adhere to the specific rules and limitations of the AI models they employ, ensuring compliance with licensing agreements and legal standards.
The legal landscape is rapidly evolving, requiring organizations to stay informed and seek legal counsel to ensure compliance.
Legal compliance is not just a regulatory hurdle. Navigating these legal implications is crucial for building trustworthy AI.


Why is data quality crucial for responsible AI? To avoid biased or discriminatory outcomes.
What is the purpose of explainable AI? To make AI decision-making transparent and understandable.
What are some key legal considerations in AI development? Data privacy, non-discrimination, intellectual property.
How can AI systems perpetuate societal biases? By being trained on biased data developed by biased individuals or deployed in biased environments.
Q:
You're developing an AI-powered loan application assessment system. Which steps would you take to ensure your system is developed responsibly? Select two.
Regularly audit the model's performance to identify and mitigate any biases that may emerge over time. and Train the model on a dataset that includes a diverse range of applicants, ensuring representation across different demographics and socioeconomic backgrounds.
Which of the following is a key aspect of securing the "model training" phase? Safeguarding training data and model parameters from unauthorized access.
What is the primary goal of the Secure AI Framework (SAIF)? To establish security standards for building and deploying AI responsibly, addressing the unique challenges and threats in the AI landscape.
What is a potential consequence of using inaccurate or incomplete data in AI training? It introduces biased outcomes and unfair results. 
What is the primary goal of ethical AI development? To ensure AI systems are used responsibly and do not cause harm.

AI as being composed of five layers. They are: infrastructure, models, platform, agents, and gen AI powered applications. 
You're likely most familiar with the gen AI powered application layer, as it's the user-facing part of generative AI, or the frontend. This is the layer that allows users to interact with and leverage the capabilities of AI. Examples of gen AI powered applications include the Gemini app, Gemini for Workspace, or NotebookLM.
Next is the agent layer. An agent is a piece of software that learns how to best achieve a goal based on inputs and tools available to it. This layer focuses on autonomous action, which describes the ability to independently set goals and carry them out within a defined environment. Agents analyze situations, use multiple tools, and make informed decisions without requiring constant human input. They are also capable of handling multi-step tasks that a model alone cannot, such as researching a topic, troubleshooting code, or accessing a system by chaining together actions. You can have a variety of agents, such as customer agents, code agents, data agents, and many more.
The platform layer typically sits between agents and models, providing the infrastructure for them to interact. For now, let's jump ahead and focus on the models themselves, and we'll come back to platforms shortly. The "brain" of the agent is the AI model. These models are complex algorithms trained on vast amounts of data. They learn patterns and relationships in the data, allowing them to generate new content, translate languages, answer questions, and much more.  
The infrastructure layer is the foundation upon which everything else rests. It provides the core computing resources needed for generative AI. This includes the physical hardware (like servers, GPUs, and TPUs) and software needed to store and run AI models and training data. 
Last, but not least, let's discuss the platform layer. The platform layer sits above the model and infrastructure layers, providing the necessary tools and infrastructure for AI development. It offers APIs, data management capabilities, and model deployment tools. This layer acts as the backbone of the system, bridging the gap between models and agents while simplifying the complexities of managing the underlying infrastructure. 

What can agents do?
Gen AI agents can process information, reason over complex concepts, and take actions. They can also be used in a variety of applications, including customer service, employee productivity, and creative tasks.

Conversational agents are designed to understand what you mean, not just what you say, and respond in a way that makes sense. 
You provide input: You can type a message or speak to the agent.
The agent understands: Using powerful AI, it figures out the meaning and intention behind your words.
The agent calls a tool: Based on your request, the agent might need to gather additional information or perform an action. This could involve searching the web, accessing a database, or interacting with another software application.
The agent generates a response: It formulates an answer that's relevant to your request and sounds natural.
The agent delivers the response: You'll see or hear the answer depending on how you interacted with it.

Workflow agents are designed to streamline your work and make sure things get done efficiently and correctly by automating tasks or going through complex processes.
You provide input: You define a task or trigger a process like submitting a form, uploading a file, initiating a scheduled event, or even ordering a product online.
The agent understands: The agent is the software that automates those steps. It interprets the task's requirements and defines the series of steps needed to complete the task.
The agent calls a tool: Based on the workflow's definition, the agent executes a series of actions. This could involve data transformation, file transfer, sending notifications, integrating with external systems, or initiating other automated processes using APIs.
The agent generates a result/output: It compiles the outcome of the executed actions, which might be a report, a data file, a confirmation message, or an updated status within a system.
The agent delivers the result/output: The agent delivers the output to the designated recipient(s) or system(s), such as via email, a dashboard, a database update, or a file storage location.

Agents incorporate two key elements that distinguish them from standalone models: a reasoning loop and tools. The reasoning loop is the agent's "thinking process." It's a continuous cycle of observing, interpreting, planning, and acting. This iterative process enables agents to analyze situations, plan actions, and adapt based on outcomes.  
Tools are functionalities that allow the agent to interact with its environment. Tools can be anything from accessing and processing data to interacting with software applications or even physical robots. This empowers agents to connect with real-world information and services, much like apps on our phones.  


🔹 GenAI-Powered Agents

Agents go a step beyond apps:

They don’t just respond → they can act on your behalf.

They combine LLMs with tools, memory, and decision-making loops.

Think of them as AI employees that can plan, reason, and execute tasks.

Examples:

Personal AI Assistants (e.g., AI email agents that read, draft, and send emails)

Business Process Automation (AI agents booking tickets, updating CRM, scheduling meetings)

DevOps Agents (AI that runs CI/CD pipelines, monitors logs, restarts failing services)

Research Agents (AI that browses the web, summarizes findings, generates reports)

At their core, agents follow a sense → think → act → learn loop

GenAI-powered applications: User-facing, task-specific (chatbots, copilots, design tools).

Agents: Goal-driven, autonomous, use reasoning + tools + memory, capable of executing multi-step tasks.

Q: 
A travel agency wants to use an AI agent to help customers plan their vacations. The agent should be able to:
• Gather customer preferences (budget, destination interests, travel dates).
• Search for flights, accommodations, and activities.
• Create personalized itineraries with options and recommendations.
• Book flights and hotels based on customer choices.
• Provide ongoing support throughout the trip.
How would the "agents" layer and the "applications" layer work together to create this AI powered travel planning experience?
The agents layer would define the AI's capabilities (searching, booking, recommending), while the applications layer would provide the user-facing tool (website or app) to interact with the agent.

A data science team is training a new generative AI model on a massive dataset of images. They need access to powerful hardware and software resources to handle the computationally intensive training process.
Which layer of the gen AI landscape would provide the necessary computational power and storage for this data science team? Infrastructure

A news organization wants to develop an AI agent that delivers personalized news to each user. The agent should be able to:
• Learn the user's interests and reading habits.
• Filter and prioritize news articles based on relevance.
• Summarize key information from multiple sources.
• Recommend related articles and diverse perspectives.
• Adapt to the user's feedback and evolving preferences.
How would the "agents" layer contribute to the functionality of this personalized news reader application?
The agents layer would define the specific tasks the AI performs, such as filtering articles, summarizing information, and making recommendations.

What are the two key elements that distinguish AI agents from standalone AI models? Reasoning loop and tools.

A game developer wants to create more realistic and engaging non-player characters (NPCs) in their game. They envision NPCs that can:
• Engage in dynamic conversations with the player.
• React to the player's actions and choices.
• Adapt their behavior based on the game's environment and storyline.
• Exhibit unique personalities and backstories.
Which layer of the gen AI landscape would be MOST crucial in defining the behaviors and capabilities of these AI powered NPCs?  
Agents

The platform layer provides the foundation for building and scaling your AI initiatives. 



Vertex AI is Google Cloud's unified machine learning (ML) platform designed to streamline the entire ML workflow. It provides the infrastructure, tools, and pre-trained models you need to build, deploy, and manage your ML and generative AI solutions.

What is an AI model?
At the heart of every AI and machine learning system lies the model. Think of it as the brain of the operation. These aren't just any algorithms. They're sophisticated mathematical structures trained on massive amounts of data. This training process allows them to learn patterns and relationships, ultimately enabling them to perform a wide range of tasks, such as generating content, analyzing data, and classifying information.
 Model Garden on Vertex AI is a service that lets you discover, customize, and deploy existing models from Google and Google partners.
Model Garden gives you options to pick from over 160 models and offers options of Google models

first party foundational models: 
Gemini Pro, Flash, and more
Imagen for text-to-image
Veo for text-to-video and image-to-video. 
Chirp 2.0 for speech-to-text.

First-party pre-trained APIs:
Build and deploy AI applications faster with our pre-trained APIs powered by the best Google AI research and technology.
Speech-to-Text specs 
Natural Language Processing (NLP) specs
Translation specs
Vision specs

Third-party models:
Model Garden supports third-party models from partners with foundation models.

open models:
Access a variety of enterprise-ready open models.
Gemma 2 specs ↗
CodeGemma specs ↗
 PaliGemma specs ↗
 Meta's Llama 3.1 specs ↗ 
 Meta's Llama 3.2 specs ↗
 Mistral AI specs ↗
 Mistral AI21specs ↗
TII's Falcon BERT, FLAN-T5, ViT, EfficientNet. View TII's Falcon specs ↗

Follow this standard workflow when creating your models in Vertex AI.
1. Gather your data, 2. Prepare your data, 3. Train, 4. Manage your model, 5. Deploy and predict

Q: 1. You need a model to translate languages for your global e-commerce platform. You want a readily available, high-performing solution and have a moderate budget.
  A:Vertex AI Model Garden
2. You're an experienced AI researcher developing a cutting-edge model for protein folding prediction. You need complete control over the model architecture and training process.
   Vertex AI - build your own custom model
3. You want to build a model to classify customer support tickets, but your team has limited machine learning expertise. You need a quick and easy solution with minimal coding.
   Vertex AI AutoML

infrastructure layer-  It's the combination of hardware and software that provides the necessary computing power, storage, and networking capabilities to train, deploy, and scale AI models.

Q:
What is the role of MLOps in Vertex AI? To automate, standardize, and manage the ML project lifecycle.
What is the purpose of AutoML in Vertex AI? 
To allow users to build and train AI models with minimal technical expertise.
What are GPUs and TPUs in the context of AI infrastructure? 
Specialized processors designed for parallel processing in AI tasks.
Which of the following statements best describes Model Garden on Vertex AI?
A service for discovering, customizing, and deploying existing AI models.
Why is high-performance storage important for generative AI? To store and efficiently access the massive datasets used in AI training.

Edge computing
While hosting infrastructure on the cloud is powerful, it's not always the ideal solution. Imagine a self-driving car needing to make split-second decisions. It can't wait for data to travel to the cloud and back. That's where edge computing comes in.
Google provides the tools to deploy your AI models in these different locations, giving you more control and flexibility.
Why go local or edge?
Imagine a drone navigating a complex environment. It needs to react instantly to obstacles, making cloud processing too slow. Running AI locally on the drone ensures real-time responsiveness. Other benefits include increased data privacy and reduced reliance on internet connectivity.
To run powerful AI models on edge devices and mobile phones, Google provides tools like Lite Runtime (LiteRT). Think of LiteRT as a platform that helps machine learning models work efficiently on your device. To learn more about Google's high-performance runtime for on-device AI, check out the article, Lite Runtime (LiteRT) Overview. 
One example of an AI model designed for edge is Gemini Nano.
Gemini Nano is Google's most efficient and compact AI model, specifically designed to run on the edge on devices like smartphones and embedded systems. It's part of the larger Gemini family of models. Think of Gemini Nano as a miniature version of the powerful AI that usually lives in Google's data centers. This "on-device" or edge approach offers several benefits.

Scenario 1
A medical device that analyzes patient data in real-time to provide immediate feedback to doctors during surgery
Edge
Real-time analysis is critical during surgery, and relying on the cloud could introduce unacceptable latency.
Scenario 2
A customer service chatbot for a large ecommerce website that handles millions of inquiries daily
Cloud
Handling millions of inquiries requires a scalable and robust cloud infrastructure
Scenario 3
A self-driving car that needs to make instant decisions to navigate traffic and avoid collisions.
Edge
Split-second decision-making in a self-driving car necessitates on-device AI processing.
Scenario 4
A system that uses AI to analyze traffic patterns and optimize traffic flow in a smart city.
Cloud
Analyzing traffic data and optimizing flow requires a centralized system with the capacity of the cloud.

Pricing for using models
Usage-based
You pay for the amount you use, often measured in tokens or characters processed. This is common for APIs like Google’s PaLM & Gemini APIs.
Subscription-based,Licensing fees, 
Pricing metrics for using models: Tokens, Characters, Requests, Compute time
Factors affecting cost : Model size and complexity, Context window, Features, Deployment

Q: You need a model that can generate realistic images from text descriptions for your new marketing campaign. You have some coding skills but are short on time. Which path would you choose? Use a pre-trained API to generate images quickly.
What is Gemini Nano designed for? Deploying AI on edge devices and smartphones
Your marketing team needs to quickly generate engaging product descriptions for an upcoming ecommerce sale. Which approach is the most efficient way to achieve this? Use a gen AI powered application like Gemini for Workspace to draft the descriptions.
Who is responsible for building and deploying custom AI agents and integrating AI capabilities into applications? Developers
What is the primary advantage of edge computing for AI applications? Real-time responsiveness and reduced latency


 Google Workspace is a collection of cloud-based productivity and collaboration tools that help people create, communicate, and collaborate, such as Gmail, Google Docs, Google Sheets, Google Meet, and Google Slides.

Google Vids is an online video creation and editing app available to Google Workspace users through your organization. 
With Help me Create in Vids, you can use Gemini to generate a first draft of your video. As a project manager, you can use Vids to share project updates, timelines, and key insights with your stakeholders to more effectively communicate information than an email or lengthy report can.
AppSheet is a no-code app development tool included with Google Workspace enterprise editions. With Gemini in AppSheet, you can quickly create apps using AI by describing your needs in a prompt using natural language.
Gemini can create an app structure with tables, columns, and links based on your description, such as "Manage inspections of my facility." You can then review and edit the app structure before creating the app in the AppSheet app editor. 

When it comes to prompting techniques, there are a few different approaches. 
Zero-shot prompting is like asking a foundation model to complete a task with no prior examples, relying solely on its existing knowledge. 
One-shot prompting involves showing the foundation model just one example, allowing it to learn and apply that knowledge to similar situations. 
Few-shot prompting, on the other hand, provides the foundation model with multiple examples to learn from, which helps it better understand the task and improve its performance.

Zero-shot prompting: A customer asks about your return policy. The LLM, drawing on its vast knowledge base, understands the question and provides a response, even though it hasn't received specific training on your company's policies.
One-shot prompting: You prompt with a single example of a customer inquiry, such as a question about shipping times. The output follows this example and accurately tags similar customer questions.
Few-shot prompting: You prompt with a few examples of customer inquiries and effective responses. The output follows these examples and accurately tags similar customer questions. For example, the output might include a discount code when a customer reports a delayed delivery, or provide detailed instructions on how to return a damaged item.

Q: You're designing a language learning app and want to incorporate an AI tutor that can provide personalized feedback to students practicing their conversational skills. Which role-prompt would be most effective in guiding the AI's interactions with students? "You are a helpful and encouraging tutor who provides constructive feedback on language learning exercises."
Which prompting technique relies entirely on the foundation model's pre-existing knowledge, without any provided examples? Zero-shot prompting
You're leading a virtual team meeting in Google Meet with colleagues located in different time zones. Some attendees are struggling to understand the discussion due to language barriers. How can Gemini assist in making this Google Meet video conference more accessible for all participants? Gemini can generate live translated captions with speaker identification, making it easier for everyone to follow along, regardless of language or accent differences.
What is the main advantage of using role prompting? 
It improves the AI model's ability to understand complex questions.
You're drafting a proposal for a new client and want to ensure your writing is clear, concise, and persuasive. How can Gemini in Google Docs assist you in this task? Gemini in Docs can provide suggestions for improving your writing style, grammar, and tone, and even generate different phrasing options to enhance your message.

Gems 
But even saved information won’t always work. Because that will apply to all your Gemini chats. Sometimes you need specific context for particular tasks or conversations. This is where Gems come in.
Creative writing: This Gem could be pre-loaded with your preferred writing style, tone, and commonly used resources. It could even help you brainstorm ideas, generate different creative text formats (poems, scripts, articles), and provide feedback on your drafts.
Coding: A coding Gem could be equipped with your favorite libraries, coding conventions, and even access to relevant documentation. It could assist with debugging, suggest code optimizations, and even generate code snippets based on your instructions.
Marketing: This Gem could be pre-loaded with your target audience demographics, competitor information, and campaign performance data. It could help you brainstorm campaign ideas, generate different marketing copy (social media posts, email newsletters, website content), analyze campaign performance, and suggest optimizations.
Think of Gems as your personalized AI assistants within Gemini.

Grounding
In the previous lesson, you learned how you can attach user-provided resources to your Gem. You can also attach documents and resources in your prompts in the Gemini app with Gemini Advanced. This is called grounding.
Grounding refers to the ability of the AI model to connect its output to verifiable and specific sources of information. 
Rag: Retrieval-augmented generation
One powerful grounding technique is retrieval-augmented generation (RAG). It involves:
Retrieving relevant information: The AI model first retrieves relevant information from a vast knowledge base (like a database, a set of documents, or even the entire web). This retrieval process is often powered by sophisticated techniques, like semantic search or vector databases.
Generating output: The model then uses this retrieved information to generate the final output. This could be anything from answering a question to writing a creative story.

NotebookLM is an AI-first notebook, grounded in your own documents, designed to help you gain insights faster. 
It’s a unique tool that is built using Gemini models and is designed to help you learn and understand information better by acting as a virtual research assistant. You can "ground" NotebookLM in specific sources like documents, presentations, or even audio and video files. This means your AI assistant will only use information from those sources to answer your questions and generate summaries.
NotebookLM is your personalized AI research partner who has read the provided source material and can answer any question about it.
NotebookLM operates on the concept of "notebooks" to help you organize and interact with your information. Here's a breakdown:
Use cases: 
Create training materials and documentation
Onboard new team members and facilitate knowledge transfer by creating a central repository for essential resources. Create a dedicated notebook to share with new hires with relevant onboarding materials (employee handbook, product documentation, training presentations). The new employees can then use NotebookLM to ask questions, clarify information, and gain a deeper understanding of their role and the company.
Researching a new topic
Gather articles and videos on a subject you're interested in and have NotebookLM create a comprehensive summary or answer specific questions you have.
Researching a new topic
Gather articles and videos on a subject you're interested in and have NotebookLM create a comprehensive summary or answer specific questions you have.
Preparing for a presentation
Feed NotebookLM your presentation slides and ask it to generate speaker notes or practice questions to help you rehearse.
Summarize documentation
Investment firms and legal teams need to thoroughly analyze documents and data. Upload financial statements, legal documents, and market research reports into NotebookLM. The team can then use the AI to quickly identify key information, summarize findings, and flag potential risks or inconsistencies.
Project proposals and plans
Keep everyone aligned and informed by creating a shared NotebookLM with all project-related information.

Q: Grounding allows an AI model to connect its responses to specific sources of information.? true
NotebookLM can access and use information from the entire internet.? false
You can use NotebookLM to create quizzes based on your uploaded documents.? true
NotebookLM is best suited for tasks like writing creative stories or poems.? false
Grounding is a feature exclusive to NotebookLM.? false
How does grounding improve the reliability of a Gem in Gemini? It connects the Gem's output to specific and verifiable sources of information.
What is the primary advantage of using Saved info in Gemini compared to reusing prompts by copying and pasting them from a document? 
Saved info ensures that specific information, like your role or company details, is consistently applied across all your Gemini interactions.
You're a freelance writer who specializes in creating different types of content, including blog posts, social media captions, and website copy. You find yourself frequently switching between different writing styles and tones depending on the project. You want to streamline your workflow and ensure consistency in your writing across various formats. Which feature of Gemini best suits your needs? Gems
What is a key advantage of using NotebookLM Enterprise for team projects? You can manage notebook access using predefined identity and access management (IAM) roles in NotebookLM Enterprise.
What is the primary function of Retrieval-Augmented Generation (RAG) in AI models? To enable the AI model to access and utilize external knowledge sources for generating outputs
You're a product manager preparing for a meeting to discuss a new product launch. You have market research reports, competitor analysis, and customer feedback surveys stored in your Google Drive. How can NotebookLM help you prepare for this meeting? NotebookLM can summarize the key findings from your documents, identify potential challenges, and provide talking points for discussion, helping you lead a productive meeting.


Gemini for Google Cloud does not use your prompts or Gemini’s responses to train models. It comes with the standard Google Cloud enterprise protections.
Gemini Cloud Assist
Gemini Cloud Assist is like having an AI expert on your team who helps you design, manage, and optimize applications on Google Cloud. It provides personalized guidance and is integrated with your Google Cloud environment to provide application lifecycle management assistance. It analyzes your cloud environment, your resources deployed, as well as metrics and logs to deliver actionable insights tailored to your needs.
Gemini in BigQuery
Gemini in BigQuery makes data analysis easier and more accessible. It can help you write code, understand your data, and even generate insights automatically, regardless of your SQL experience. This means faster and more efficient data exploration.
Gemini Code Assist
Gemini Code Assist acts as an AI pair programmer, helping developers write better code, just like when two programmers work together to solve problems. Gemini Code Assist can provide code suggestions, generate code blocks, and even offer explanations. It supports over 20 popular programming languages, code editors, and developer platforms, all of which help developers increase productivity.  
Gemini in Colab Enterprise
A Colab Enterprise notebook is an interactive environment that lets you write and execute code. Gemini in Colab Enterprise can use AI to help you write Python code in your notebook by suggesting code segments as you type and generating code based on your descriptions. This streamlines your data analysis and machine learning workflows.
Gemini in Databases
Gemini in Databases helps developers and database administrators manage their databases more effectively. It uses AI to simplify many aspects of using a database, from building applications with natural language to managing an entire fleet of databases from a single interface.
Gemini in Looker
With Gemini in Looker, you can analyze data and gain insights faster. As an intelligent assistant, it helps you understand your data, create visualizations, and even generate reports, making data exploration more intuitive.
Gemini in Security
Gemini in Security helps security teams detect, contain, and stop threats from spreading. It provides near-instant analysis of security findings and potential attack paths. Gemini in Security also summarizes prevalent tactics, techniques, and procedures used by threat actors, giving customers around the world detailed and timely threat intelligence.

Q:
Which of the following statements about the Gemini for Google Cloud enterprise security model is true? 
Customer data is encrypted and access controls are in place.
A team of developers wants to improve their coding efficiency and collaboration. Which Gemini for Google Cloud tool would be most beneficial for them? Gemini Code Assist

There are generally two kinds of agents: deterministic and generative. So those agents that you described can be referred to as traditional or deterministic agents.
A deterministic agent is an agent that is based on predefined paths and actions. It is typically workflow-based and event-driven, and it offers a high degree of control and predictability.

components of an agent:
Foundational model:This is the underlying language model (LLM) that powers the agent. It could be a small or large language model, a Google model like Gemini, or a model from another provider. The key is to select a model with training data relevant to the agent's intended use case.
Tools: Tools enable the agent to interact with the outside world. These can include extensions that connect to APIs, functions that act as mock API calls, and data stores like vector databases. Tools allow the agent to not only observe the world but also act upon it.
Reasoning loop: This is the core of the agent, responsible for making decisions and taking actions. It's an iterative process where the agent considers its goal, the available tools, and the information it has gathered. Frameworks like ReAct (Reason and Act) are commonly used to guide the reasoning process.

Sampling parameters and settings:
Token count:
Imagine each word and punctuation mark in your text as a character. These characters are grouped into smaller units called tokens, which represent meaningful chunks of text. Models have limits on how many tokens they can handle at once. A higher token count allows for longer and more complex conversations, but it also requires more processing power. For example, one token is roughly equivalent to four characters in English. So, a hundred tokens would be about sixty to eighty words.
Temperature:This parameter controls the "creativity" of the model, because it adjusts the randomness of word choices during text generation, influencing the diversity and unpredictability of the output. A higher temperature makes the output more random and unpredictable, while a lower temperature makes it more focused, deterministic and repeatable.
Top-p (nucleus sampling):
"Top-p" stands for the cumulative probability of the most likely tokens considered during text generation. This is another way to control the randomness of the model's output. It concentrates the probability on the most likely tokens, making the output more coherent and relevant. A lower top-p value leads to more focused responses (i.e. only the most probable tokens), while a higher value allows for more diversity (i.e. extending to lower probability tokens as well).
Safety settings:These settings allow you to filter out potentially harmful or inappropriate content from the model's output. You can adjust the level of filtering based on your specific needs and preferences.
Output length:This determines the maximum length of the generated text. You can set it to a specific number of words or characters or allow the model to generate text until it reaches a natural stopping point.
By experimenting with these parameters, you can significantly influence the AI model's behavior. For example, if you need a concise and factual answer, you might use a lower temperature and a smaller output length. If you're looking for a more creative and open-ended response, you could increase the temperature and top-p values.

Google AI Studio:
Google AI Studio is a web-based tool that allows developers, students, and researchers to try Gemini models and begin building with the Gemini Developer API. It is designed for ease of use and accessibility, targeting a broad audience, including non-technical users who want to leverage AI capabilities without deep expertise in machine learning.
Vertex AI Studio:
Vertex AI Studio is a Google Cloud console tool for rapidly prototyping and testing generative AI models. It provides developers with a space to test models using prompt samples, design and save prompts, and tune foundation models.

Attributes	Google AI Studio	Vertex AI Studio
Focus	A streamlined easy-to-use interface for exploring Gemini's capabilities, experimenting with parameters, and generating different creative text formats. 	A comprehensive and robust environment for building, training, and deploying machine learning models that is part of the Google Cloud Vertex AI Platform.
Users	Beginners, hobbyists, and those at the initial stages of development.	    Professionals, researchers, and developers.
Access	Login with a standard Google account.	Access through Google Cloud.
Limitations	It has usage limits (queries per minute, requests per minute, tokens per minute), making it less suitable for large-scale or production-level applications. It's primarily intended for initial prototyping or small-scale model deployments	Has a service charge based on your usage.
Advantages	A simplified interface that is easy to get started and use, even for those without deep machine learning expertise.	More flexible quotas for usage, potentially increased upon request. Offers enterprise-grade security and compliance features.
Choosing the right tool depends on your specific needs and expertise. If you're just starting out, Google AI Studio is a great place to learn and experiment. If you need a more powerful and scalable solution for professional use cases, Vertex AI Studio is the better choice.

The reasoning loop:-
The reasoning loop is a key component of a generative AI agent that governs how the agent takes in information, performs internal reasoning, and uses that reasoning to inform its next action or decision. It is an iterative, introspective process that continues until the agent achieves its goal or reaches a stopping point. The complexity of the reasoning loop can vary greatly depending on the agent and the task it is performing.
Iterative process
The reasoning loop is not a one-time operation, but rather a cyclical process where the agent continuously evaluates its progress and determines the next best action to take. This loop involves steps for action, tool selection, and observation.
Internal reasoning
The agent uses its underlying language model to think through the steps it needs to take to complete a task. The language model provides the agent with reasoning and logic capabilities.Decision making
Based on its internal reasoning, the agent decides on the next course of action. This involves choosing the appropriate tools to use and determining the necessary inputs for those tools.Reasoning frameworks
The reasoning loop utilizes various prompt engineering frameworks and techniques to guide its reasoning and planning.
Prompt engineering techniques:
There are many different prompting techniques that one can use, but we will highlight two of the most common for this course: ReAct prompting and chain-of-thought (CoT) prompting.
ReAct is a prompting framework that allows the language model to reason and take action on a user query, with or without in-context examples.Chain-of-thought prompting is a technique where you guide a language model through a problem-solving process by providing examples with intermediate reasoning steps, helping it learn to approach new problems in a more structured and logical way.

Key components of ReAct
Think: The LLM generates a thought about the problem, similar to CoT.
Act: The LLM decides what action to take, such as searching the web, accessing a database, or using a specific tool; the LLM specifies the input for the action, like a search query or database command.
Observe: The LLM receives feedback from the action, such as search results or database entries.
Respond: The LLM generates a response, which could involve providing an answer to the user, taking further actions, or formulating a new thought for the next iteration.

Chain-of-thought (CoT)
Remember prompt chaining? Where you keep prompting in the same thread so the LLM keeps your chat history and learns more as you go? Well, you can do something very similar behind the scenes for your user using chain-of-thought (CoT) prompting. CoT enables reasoning capabilities through intermediate steps. 
Think of CoT as a way to make LLMs even smarter by teaching them to think step-by-step, just like a human would. Instead of just giving the LLM a prompt and expecting an answer, you guide it through the reasoning process. You provide examples of how to solve similar problems, showing the steps involved. This helps the LLM learn to approach problems in a more logical and structured way. It is similar to teaching a student to think out loud. You're teaching the LLM to think out loud. By showing it the intermediate steps, you're helping it develop a "chain of thought" that leads to the correct answer.
Why is CoT important?
CoT is a powerful technique that helps LLMs think a bit more like humans. By guiding them through the reasoning process, we can unlock their full potential and achieve even more impressive results.
Improved reasoning: CoT helps LLMs solve complex problems that require logical thinking.
Better accuracy: By breaking down problems into smaller steps, CoT can lead to more accurate results.
Enhanced explainability: CoT makes it easier to understand how the LLM arrived at an answer, which is important for building trust and transparency.
Key components of CoT
Just like there are different ways to solve a problem, there are different ways to implement CoT. Some popular techniques include:
Self-consistency: Encouraging the LLM to generate multiple solutions and choose the most consistent one.
Active-prompting: Allowing the LLM to ask clarifying questions or request additional information.
Multimodal CoT: Combining text with other forms of data, like images or videos, to enhance reasoning.

CoT focuses on internal reasoning, guiding the LLM through a chain of thought.

ReAct focuses on external interaction, allowing the LLM to gather information and take actions in the real world.

Types of agent tools
Agent tooling equips agents with the resources they need to be effective. Think of it as providing the agent with the right skills, connections, and knowledge to achieve its goals. These tools allow agents to access information, perform actions, and interact with various systems.
Extensions (APIs)
Extensions bridge the gap between an agent and external APIs. APIs (application programming interfaces) are sets of rules that govern how software interacts. Extensions provide a standardized way for agents to use APIs, regardless of the API's specific design. This simplifies API interaction, making it easier for agents to access external services and data.
Example: An agent designed to book travel might use an extension to interact with a travel company’s API. The extension handles the complexities of communicating with travel company’s systems, allowing the agent to focus on the task of finding and booking flights.
Functions
Functions are like specialized tools within the agent's toolbox. They represent specific actions the agent can perform. An agent's reasoning system selects the appropriate function based on the task at hand. Functions can encapsulate complex logic or interactions, making them reusable and manageable.
Example: A "calculate_price" function might take flight details and passenger information as input and return the total cost. The agent can call this function whenever it needs to calculate a price.
Data stores
Data stores provide agents with access to information. This can include real-time data, historical data, or knowledge bases. Data stores ensure that the agent's responses are accurate, relevant, and up-to-date.
Example: An agent might use a data store to access current weather conditions, stock prices, or a database of customer information.
Plugins
Plugins extend the agent's capabilities by adding new skills or integrations. They can connect the agent to specific services, provide access to specialized tools, or enable interaction with particular platforms.
Example: A plugin could enable an agent to interact with a calendar application, allowing it to schedule appointments. Another plugin might integrate with a payment gateway, enabling the agent to process transactions.

How the reasoning loop works with tools:
1. Reasoning (tool selection): The agent analyzes the task and determines which tools are needed. It considers the available extensions, functions, and data stores to choose the most appropriate resources.
2. Acting (tool execution): The agent executes the selected tool. This might involve calling an API via an extension, invoking a function, or querying a data store. The agent provides the necessary inputs to the tool.
3. Observation: The agent receives the output from the tool. This output becomes the "observation" in the ReAct cycle.
4. Iteration (dynamic iteration): Based on the observation, the agent reasons about the next steps. It might need to select different tools, refine its approach, or gather more information. This cycle repeats until the task is complete.

Key Google Cloud tool for agents:
Cloud Storage
A highly scalable and durable object storage service. Use Cloud Storage to store and retrieve data that your agent needs.
Databases (Cloud SQL, Cloud Spanner, Firestore)
Google Cloud offers a variety of database solutions to suit your needs. Your agent can use these databases to store and retrieve information, manage user data, or track its own progress.
Cloud Run functions
Create serverless functions that act as specialized tools for your agent. Cloud Run functions can be used to connect to databases, call external APIs, perform complex calculations, or handle other specific tasks. They are easily triggered by your agent and scale automatically.
Cloud Run
For more complex agent tools that require containerized environments, Cloud Run  provides a serverless platform for deploying and running stateless containers. This is ideal for custom tools that have specific dependencies or require more control.
Vertex AI
Agents can use other agents as tooling. You can use Vertex AI to create models or agents that are called as tooling by other agents.

To actually use the API within your applications, you'll need to generate an API key for Google AI Studio or set up authentication and authorization for Vertex AI Studio. 
You can also leverage your API or code through some of Google’s no code and low code tooling, such as:
Apps Script
A cloud-based platform that lets you automate Google Workspace tasks by leveraging a combination of JavaScript code and Google's built-in services. You can use the Gemini API with Apps Script to build add-ons that enhance features in Google Workspace using generative AI.
AppSheet
AppSheet is a no-code platform for building custom business apps. You can use Apps Script to extend the capabilities of your AppSheet applications using custom logic or by integrating APIs. With the Gemini API, for example, you can automate tasks, generate content, or provide intelligent assistance directly within your AppSheet app.
​Multi-agent applications​
When leveraging agents into your application, consider situations when you may need multiple agents.These systems utilize multiple agents, each potentially specialized for a specific task, to create more efficient and capable applications. For instance, a travel booking app might use one agent to search for flights, another to find hotels, and a third to suggest local attractions.These agents can work independently or interact with each other to provide a seamless booking experience. This modular approach not only improves efficiency but also allows for greater flexibility and scalability. Even more interestingly, an agent itself can be a tool within another agent. A sentiment analysis agent, for example, could be used by a customer service agent to gauge user satisfaction during an interaction.

Q: Which parameter in Google AI Studio would you adjust to control the randomness and creativity of Gemini's output? temperature
Which of the following statements best describes the function of a "reasoning loop" within a generative AI agent? 
It guides the agent through an iterative process of observation, internal reasoning, and decision-making to solve tasks.
Imagine an AI agent tasked with scheduling a team meeting. The agent needs to access the team members' calendars, propose suitable time slots, and send out invitations. Which combination of agent tools would be most essential for this task? Extensions and plugins
What is the primary distinction between Google AI Studio and Vertex AI Studio in terms of their target users? Google AI Studio is designed for beginners and experimentation, while Vertex AI Studio caters to professionals and larger-scale projects.
Which of the following scenarios best illustrates the concept of a multi-agent system in the context of generative AI?
A system where multiple specialized AI agents collaborate to achieve a complex goal, such as booking a trip.


Pre-RAG history
Before retrieval-augmented generation (RAG), models could not directly learn from tool data. They could use tools to fetch information, but lacked the ability to process and integrate that data into their knowledge. Their understanding was limited to their training data, and tool-retrieved data was used only for the immediate query, not for learning. RAG solves this by enabling models to retrieve and learn from data provided by tools.
Retrieval-augmented generation (RAG) enhances the capabilities of large language models (LLMs) by grounding their responses in external knowledge sources. This process allows the model to access and process information beyond its training data, leading to more accurate, relevant, and up-to-date responses.
After the user submits a query or request to the LLM, here's how RAG works with tools.
Retrieval:The LLM, equipped with retrieval tools, identifies relevant information from external sources. These tools can include:
Data stores: These can be internal databases or other sources of structured and unstructured data.
Vector databases: These databases store embeddings (numerical representations) of text, allowing the LLM to find semantically similar information to the user's query. The LLM uses the user query to create an embedding and searches the vector database for matching documents or passages.
Search engines: The LLM can use search engines (via extensions or APIs) to find relevant web pages, articles, or other online content.
Knowledge graphs: These structured databases store information about entities and their relationships. The LLM can query knowledge graphs to retrieve facts and relationships relevant to the user's query.
Augmentation: The retrieved information is then incorporated (or "augmented") into the prompt that is fed to the LLM. This augmented prompt now contains both the user's original query and the relevant context retrieved from external sources.
Generation: The LLM processes the augmented prompt and generates a response. Because the prompt includes relevant external information, the LLM can generate a more informed, accurate, and contextually appropriate response. It can also cite the sources of its information, increasing transparency and trustworthiness.

Vertex AI Search:Vertex AI Search offers both search and recommendation solutions.
Vertex AI Search combines information retrieval, natural language processing, and large language models to understand what people are really asking, even if they phrase it poorly.
Search also extends to include some specific forms of search such as: Document search, Media search, Healthcare search, Search for commerce.

How Vertex AI Search works
Regardless of the specific search or recommendation option you choose, Vertex AI Search operates on a foundation of intelligent data connection, grounding, and generative AI. It seamlessly connects to your existing data stores, whether they are structured databases, unstructured document repositories, or a combination of both. This connection is crucial as it allows Vertex AI Search to act as an agent, observing the user's query or context (the environment) and acting by retrieving relevant information or suggesting relevant items (using the data stores as tools) to achieve the goal of providing the right information or recommendation at the right time.
A key strength of Vertex AI Search lies in its ability to ground gen AI LLM responses with your first-party data, curated third-party data, and even Google's knowledge graph (Grounding with Google Search), minimizing "hallucinations" and ensuring trustworthy information.
This grounding is where the connection to RAG comes in. By using your own data sources as the foundation for LLM responses, Vertex AI Search implements a RAG approach, ensuring that the information provided is relevant, accurate, and grounded in the context of your specific data.

Conversational Agents: Conversational Agents to act as effective chatbots communicating with your customers.
Agent Assist: Agent Assist to support your live human contact center agents.
Conversational Insights: Conversational Insights to gain insights into all your communications with customers (through chatbot agents or human agents).
All of this can be built on top of Google’s Contact Center as a Service (CCaaS), an enterprise-grade contact center solution that is native to the cloud. Let’s dive deeper into the details of how these all work and where generative AI fits in.

Deterministic:Deterministic is more associated with some historical agents. It is a rule-based, very defined system for your chatbot agents to follow. It will use very defined logic such as if the user presses this number, go to this route. Everything you want a deterministic agent to do needs to be explicitly defined. Deterministic agents usually require low to medium code to build.
Generative: Generative is based on new generative AI technology. It uses large language models to give a real conversational feel to your chatbot. These agents will determine what to do on their own based on your prompt unless you specifically tell it otherwise. Generative agents usually require prompting and either no code or low code to build.
Hybrid: Google Cloud Conversational Agents empowers you to build hybrid agents that combine the strengths of deterministic and generative AI. This approach offers strict control while leveraging generative AI's flexibility to better address customer needs

Agent Assist
There will always be use cases where conversational agents are not enough or tricky situations where the human touch is needed. This is where live agents come in. But using human agents comes with its own challenges. Not all human agents are the same and they have different levels of experience. There can be a lot of training needed, especially when a new agent starts.
This is where Agent Assist comes in. It is a tool that supports live human agents with in-the-moment assistance, generated responses, and real-time coaching to help them resolve customer issues faster and with greater accuracy. Using AI and generative AI, Agent Assist can recommend agent responses to customers, suggest the appropriate knowledge base content to solve a customer’s issue, transcribe or translate calls in real time, summarize conversations and more.

Creating playbooks
When building a generative AI agent with Conversational Agents, you create what is called a playbook for how you want your agent to act. In the playbook, you define your agent's goal, such as providing customer support, answering user questions, or even generating creative content. You then provide detailed instructions on how the agent should act and any rules you would like it to follow. In the playbook, you also have the option to link it to external tools such as data stores. Once your playbook is defined, you are ready to go and can start testing your agent and interacting with it.

Metaprompting is about creating prompts that guide the AI to generate, modify, or interpret other prompts.

Google Agentspace is designed to help you and your team use your company's information more effectively. It uses AI to create customized agents that can access and understand data from various sources, regardless of where that data is stored. These agents can then be integrated into your organization’s internal websites or dashboards. Think of it as a way to give your employees their own personal research assistants, but for work.
Connect to NotebookLM Enterprise
Connect to NotebookLM Enterprise so employees can upload information to analyze, get insights, and listen to audio summaries of data.
Connect to multimodal search agents
Connect to multimodal search agents grounded in your data (structured and unstructured) across multiple systems (Google Drive, Confluence, Jira, ServiceNow, and more) so your employees can find relevant information within your company.
Add generative AI assistants
Add generative AI assistants that are grounded in your enterprise data and can be prompted to take actions through connectors (tools). For example, creating Google Calendar events or editing issues in Jira Cloud.
Add custom agents
Add custom agents through Conversational Agents.
How is Agentspace different from NotebookLM? 
While Agentspace can connect to NotebookLM Enterprise, they serve different core purposes. Think of NotebookLM Enterprise as your specialized AI tool for diving deep into specific documents and web sources – asking questions, summarizing, and creating new content based only on those sources. Agentspace, on the other hand, is your comprehensive enterprise AI assistant. It uses AI agents and unified search to automate tasks and find information across all your connected business systems, not just specific documents you upload.

Plan for generative AI integration:
Establish a clear vision
Define how gen AI integrates with your overall corporate strategy. Secure buy-in from leadership and appoint a champion to drive adoption.
Prioritize high-impact use cases
Focus on opportunities that align with business goals, considering factors like technical feasibility, data readiness, and potential for transformation.
Invest in capabilities
Develop the necessary skills within your organization through a combination of upskilling, recruitment, and training. Foster a culture of learning and experimentation to empower teams to leverage gen AI effectively.
Drive organizational change
Proactively manage the shifts in workflows, roles, and mindsets that accompany gen AI implementation. Encourage agile collaboration and knowledge sharing to scale best practices.
Measure and demonstrate value
Track, measure, and assess the impact of gen AI initiatives. Use data-driven insights to continuously improve and demonstrate the value of your efforts.
Champion responsible AI
Develop and implement a robust framework for responsible AI, ensuring fairness, transparency, privacy, and security in all generative AI initiatives.

Plan for change:
Regularly review and refine your strategy based on the latest advancements and your organization's specific needs.  

Stay informed about new models, tools, and techniques by actively following industry news, research publications, and expert opinions.  

Engage with the generative AI community through conferences, workshops, and online forums to learn from others and share your knowledge. 

Invest in training and development programs to upskill your workforce and equip them with the knowledge and skills needed to thrive in the generative AI era.  

Attract and retain top talent by creating a stimulating and rewarding work environment that fosters creativity and professional growth.  

Q:
According to the provided information, how does a large language model (LLM) interact with data stores in a retrieval-augmented generation (RAG) workflow?The LLM queries data stores to retrieve information relevant to a user's request, enriching its response.
Which of the following options best identifies the key components of an AI agent? A reasoning loop, tools, and a foundational model.
What are the key steps in planning for impact when implementing generative AI solutions? 
Define key metrics, collect and analyze data, and iterate and improve.
What is the primary purpose of Google Agentspace within an organization? To provide a centralized platform for organizing and accessing AI agents that can utilize company data to assist employees.
Which core capability of Vertex AI Search helps mitigate the issue of hallucinations in generative AI, ensuring that search results are grounded in reliable information? Grounding
What is a key advantage of using a hybrid approach (combining deterministic and generative AI) when building conversational agents with Google Cloud's Customer Engagement Suite?
It enables strict control over agent behavior while leveraging the flexibility of generative AI.

