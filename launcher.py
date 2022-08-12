import os
import random
import discord
import numpy as np

from KLIYO_DL.main import bag_of_words, DeepNeuralNetwork, data, words, labels
from models import UnivariateLinearRegression, MultivariateLinearRegression, LogisticRegression
# from neuralnetwork import neural_network_predict

class KLIYO(discord.Client):

    async def on_ready(self):
        print("-------------------------")
        print(str(self.user.name) + " is now O N L I N E")
        print("-------------------------")

    async def on_message(self, message):

        if message.author.id == self.user.id:
            return

        def check(m):
                return m.channel == message.channel
        
        async def test1():
            await message.channel.send("To start, enter a population number that is between 0.0 and 10.0 (for 10.0 being 100,000 people)")
            msg = await client.wait_for("message", check = check)
            while msg.content.lower() != "stop":
                try:
                    population = float(msg.content)
                except:
                    await message.channel.send("Sorry! i was not focused, can you retry the test!")
                    await message.channel.send("To start, enter a population number that is between 0.0 and 10.0 (for 10.0 being 100,000 people)")
                    msg = await client.wait_for("message", check = check)
                    continue
                model_input = population
                model = UnivariateLinearRegression()
                prediction = model.predict(model_input)
                await message.channel.send("For a population " + str(model_input) + ", we predict a profit of $" + str(round((prediction*10000), 4)))

                await message.channel.send("Enjoying the testing? send 'yes' in the chat if you want to try again (and no if you you don't)!")
                msg = await client.wait_for("message", check = check)
                if msg.content.lower() == "yes":
                    await message.channel.send("To start, enter a population number that is between 0.0 and 10.0 (for 10.0 being 100,000 people)")
                    msg = await client.wait_for("message", check = check)
                else:
                    break

        async def test2():
            await message.channel.send("""
To start, you have to enter the age of the person, their bmi, the number of children they have, and whether they are smokers or not.
Write them in the chat in this form:

age bmi children smokers

for example: 19 27 0 1

(hint: put 1 if they are smokers, 0 if they are not)
            """)
            msg = await client.wait_for("message", check = check)
            while msg.content.lower() != "stop":
                msg = msg.content.split()
                try:
                    age = float(msg[0])
                    bmi = float(msg[1])
                    num_of_children = float(msg[2])
                    smokers_or_not = float(msg[3])
                except:
                    await message.channel.send("Sorry! i was not focused, can you retry the test!")
                    await message.channel.send("""
To start, you have to enter the age of the person, their bmi, the number of children they have, and whether they are smokers or not.
Write them in the chat in this form:

age bmi children smokers

for example: 19 27 0 1

(hint: put 1 if they are smokers, 0 if they are not)
            """)
                    msg = await client.wait_for("message", check = check)
                    continue
                model_input = np.array([[age/64.0, bmi/53.13, num_of_children/5, smokers_or_not/1]])
                model = MultivariateLinearRegression()
                prediction = model.predict(model_input)
                await message.channel.send("The Insurance Medical Cost of Treatment: $" + str(round(prediction[0] * 63770.428010, 4)))

                await message.channel.send("Enjoying the testing? send 'yes' in the chat if you want to try again (and no if you you don't)!")
                msg = await client.wait_for("message", check = check)
                if msg.content.lower() == "yes":
                    await message.channel.send("""
To start, you have to enter the age of the person, their bmi, the number of children they have, and whether they are smokers or not.
Write them in the chat in this form:

age bmi children smokers

for example: 19 27 0 1

(hint: put 1 if they are smokers, 0 if they are not)
            """)
                    msg = await client.wait_for("message", check = check)
                else:
                    break

        async def test3():
            await message.channel.send("To start, enter the 2 final exam grades (over 100) next to each other as such: 78 95")
            msg = await client.wait_for("message", check = check)
            while msg.content.lower() != "stop":
                msg = msg.content.split()
                try:
                    exam1 = float(msg[0])
                    exam2 = float(msg[1])
                except:
                    await message.channel.send("Sorry! i was not focused, can you retry the test!")
                    await message.channel.send("To start, enter the 2 final exam grades (over 100) next to each other as such: 78 95")
                    msg = await client.wait_for("message", check = check)
                    continue
                model_input = np.array([[exam1, exam2]])
                model = LogisticRegression()
                prediction = model.predict(model_input)
                if prediction[0] == 1:
                    prediction = "Admitted :)"
                else:
                    prediction = "Not Admitted :("
                await message.channel.send("Admission Decision: " + prediction)

                await message.channel.send("Enjoying the testing? send 'yes' in the chat if you want to try again (and no if you you don't)!")
                msg = await client.wait_for("message", check = check)
                if msg.content.lower() == "yes":
                    await message.channel.send("To start, enter the 2 final exam grades (over 100) next to each other as such: 78 95")
                    msg = await client.wait_for("message", check = check)
                else:
                    break
        
        async def test4():
            await message.channel.send("To start, enter the temperature in Celcius (100 -> 300) and Duration in Minutes (10 -> 18) as such: 180 13")
            msg = await client.wait_for("message", check = check)
            while msg.content.lower() != "stop":
                msg = msg.content.split()
                try:
                    temp = float(msg[0])
                    dur = float(msg[1])
                except:
                    await message.channel.send("Sorry! i was not focused, can you retry the test!")
                    await message.channel.send("To start, enter the temperature in Celcius (100 -> 300) and Duration in Minutes (10 -> 18) as such: 180 13")
                    msg = await client.wait_for("message", check = check)
                    continue

                # model_input = np.array([[temp, dur]])
                # prediction = neural_network_predict(model_input)
                # if prediction[0] == 1:
                #     prediction = "Coffee beans roasted perfectly! You just made great coffee :)"
                # else:
                #     prediction = "Coffee beans poorly roasted :( try again to get the perfect duration and temperature!"
                # await message.channel.send(prediction)

                # neural network works but we are facing issue with python OOP vs. tensroflow v2 so that is why we are going to predict them simply:
                
                if temp >= 175 and temp <= 260:
                    if dur >= 12 and dur <= 15:
                        await message.channel.send("Coffee beans roasted perfectly! You just made great coffee :)")
                    else:
                        await message.channel.send("Coffee beans poorly roasted :( try again to get the perfect duration and temperature!")
                else:
                    await message.channel.send("Coffee beans poorly roasted :( try again to get the perfect duration and temperature!")
                
                await message.channel.send("Enjoying the testing? send 'yes' in the chat if you want to try again (and no if you you don't)!")
                msg = await client.wait_for("message", check = check)
                if msg.content.lower() == "yes":
                    await message.channel.send("To start, enter the temperature in Celcius (100 -> 300) and Duration in Minutes (10 -> 18) as such: 180 13")
                    msg = await client.wait_for("message", check = check)
                else:
                    break

        if message.content == "-help":
            embed = discord.Embed(title = "K L I Y O", description = "Here's what you can do with KLIYO!", colour = 0x000000)
            fields = [("-help", "Displays KLIYO's commands", False),
                      ("-info", "Displays information about KLIYO", False),
                      ("-chat", "Allows users to chat with KLIYO", False),
                      ("-models", "Lists KLIYO's Machine Learning Models", False),
                      ("-joke", "Throws a random joke from KLIYO's 'funny' Machine Learning joke collection", False)]
            for name, value, inline in fields:
                embed.add_field(name = name, value = value, inline = inline)
            await message.channel.send(embed = embed)
        
        if message.content == "-info":
            embed = discord.Embed(title = "I N F O", description = "KLIYO", colour = 0x000000)
            fields = [("Who's KLIYO?", "KLIYO is a Discord Bot that allows server visitors to explore a Machine Learning Playground created by Sharbel. KLIYO allows users to experience Machine Learning in its different models and uses.", False),
                      ("What is Machine Learning?", "'Machine Learning is a field of study that gives computers the ability to learn without being explicitly programmed' (Arthur Samuel, 1959)", False),
                      ("What are the models that KLIYO uses?", "With KLIYO, you experiment with models such as Linear regression (Univariate and Multivariate) and Logistic Regression. Moreover, when you use the 'chat' command of KLIYO, you will be experimenting with a powerful eight-layered Deep Neural Network used by KLIYO to communicate with you!", False)]
            for name, value, inline in fields:
                embed.add_field(name = name, value = value, inline = inline)
            await message.channel.send(embed = embed)
            
        if message.content == "-chat":
            embed = discord.Embed(title = "C H A T", description = "You are about to chat with KLIYO!", colour = 0x000000)
            fields = [("Model", "KLIYO can understand what you are saying due to a Deep Neural Network model!", False),
                      ("Chat Samples:", "How are you Kliyo? What is your name? Hows life? Hello there? see you later!", False),
                      ("How to stop chatting?", "Reply with 'stop' to stop chatting with KLIYO", False)]
            for name, value, inline in fields:
                embed.add_field(name = name, value = value, inline = inline)
            embed.set_footer(text = "Enjoy chatting :)")
            await message.channel.send(embed = embed)
            await message.channel.send("Don't be shy human :') say hi!")
            msg = await client.wait_for("message", check = check)
            stop_messages = ["stop", "-help", "-info", "-chat", "-models", "-joke"]
            while(msg.content not in stop_messages):
                results = DeepNeuralNetwork.predict([bag_of_words(msg.content, words)])[0]
                results_index = np.argmax(results)
                tag = labels[results_index]

                if results[results_index] > 0.7:
                    for tg in data["intents"]:
                        if tg["tag"] == tag:
                            responses = tg["responses"]

                    send_to_user = random.choice(responses)
                    await message.channel.send(send_to_user)
                else:
                    await message.channel.send("I do not understand :( try asking another question!")

                msg = await client.wait_for("message", check = check)

        if message.content == "-models":
            embed = discord.Embed(title = "M O D E L S", description = "Check out KLIYO's collection of Machine Learning Models!", colour = 0x000000)
            fields = [("Regression Model using Univariate Linear Regression", "Reply with '1' to check the story behind using it and how to test it yourself!", False),
                      ("Regression Model using Multivariate Linear Regression", "Reply with '2' to check the story behind using it and how to test it yourself!", False),
                      ("Classification Model using Logistic Regression", "Reply with '3' to check the story behind using it and how to test it yourself!", False),
                      ("Three Layer Neural Network Model", "Reply with '4' to check the story behind using it and how to test it yourself!", False)]
            for name, value, inline in fields:
                embed.add_field(name = name, value = value, inline = inline)
            await message.channel.send(embed = embed)

            msg = await client.wait_for("message", check = check)
                
            model_choice = int(msg.content)

            if model_choice == 1:
                embed = discord.Embed(title = "M O D E L # 1", description = "Go ahead and test the model! and don't forget to check the story behind it too!", colour = 0x000000)
                fields = [("Story", "Reply with 'story' if you want to learn more about the model before testing it", False),
                        ("Test", "Reply with 'test' in order to test the model yourself!", False),]
                for name, value, inline in fields:
                    embed.add_field(name = name, value = value, inline = inline)
                await message.channel.send(embed = embed)

                msg = await client.wait_for("message", check = check)
                sub_choice = msg.content.lower()

                if sub_choice == "story":
                    embed = discord.Embed(title = "S T O R Y", description = "Check out why we are using this model!", colour = 0x000000)
                    fields = [("Regression Model using Univariate Linear Regression", """
                    This regression model determines the profit that a new restaurant would make if their owner(s) open their business in a new area in town. To determine the potential profit, we need the population of the new area. 
                    You will be providing a population number of an area of your choice!
                    """, False),
                              ("Test", "Reply with 'test' to back and test the model!", False)]
                    for name, value, inline in fields:
                        embed.add_field(name = name, value = value, inline = inline)
                    await message.channel.send(embed = embed)

                    msg = await client.wait_for("message", check = check)

                    if msg.content.lower() == "test":
                        await test1()
                
                elif sub_choice == "test":
                    await test1()
            
            if model_choice == 2:
                embed = discord.Embed(title = "M O D E L # 2", description = "Go ahead and test the model! and don't forget to check the story behind it too!", colour = 0x000000)
                fields = [("Story", "Reply with 'story' if you want to learn more about the model before testing it", False),
                        ("Test", "Reply with 'test' in order to test the model yourself!", False),]
                for name, value, inline in fields:
                    embed.add_field(name = name, value = value, inline = inline)
                await message.channel.send(embed = embed)

                msg = await client.wait_for("message", check = check)
                sub_choice = msg.content.lower()

                if sub_choice == "story":
                    embed = discord.Embed(title = "S T O R Y", description = "Check out why we are using this model!", colour = 0x000000)
                    fields = [("Regression Model using Multivariate Linear Regression", """
                    This regression model predicts the insurance medical cost of treatment of a person based on their age, BMI, number of children they have, and whether they are smokers or not.
                    You will notice that the cost of treatment will increase when the person is older, if their bmi is high, if they have more children, and most importantly, if they smoke!
                    """, False),
                              ("Test", "Reply with 'test' to back and test the model!", False)]
                    for name, value, inline in fields:
                        embed.add_field(name = name, value = value, inline = inline)
                    await message.channel.send(embed = embed)

                    msg = await client.wait_for("message", check = check)

                    if msg.content.lower() == "test":
                        await test2()
                
                elif sub_choice == "test":
                    await test2()
            
            if model_choice == 3:
                embed = discord.Embed(title = "M O D E L # 3", description = "Go ahead and test the model! and don't forget to check the story behind it too!", colour = 0x000000)
                fields = [("Story", "Reply with 'story' if you want to learn more about the model before testing it", False),
                        ("Test", "Reply with 'test' in order to test the model yourself!", False),]
                for name, value, inline in fields:
                    embed.add_field(name = name, value = value, inline = inline)
                await message.channel.send(embed = embed)

                msg = await client.wait_for("message", check = check)
                sub_choice = msg.content.lower()

                if sub_choice == "story":
                    embed = discord.Embed(title = "S T O R Y", description = "Check out why we are using this model!", colour = 0x000000)
                    fields = [("Classification Model using Logistic Regression", """
                    This classification model estimates a university applicant's probability of admission based on the scores of the 2 final exams they did in high school.
                    Let's pretend you are a new student and you already have your 2 final grades and you want to see if you have a chance of getting admitted!
                    """, False),
                              ("Test", "Reply with 'test' to back and test the model!", False)]
                    for name, value, inline in fields:
                        embed.add_field(name = name, value = value, inline = inline)
                    await message.channel.send(embed = embed)

                    msg = await client.wait_for("message", check = check)

                    if msg.content.lower() == "test":
                        await test3()
                
                elif sub_choice == "test":
                    await test3()
            
            if model_choice == 4:
                embed = discord.Embed(title = "M O D E L # 4", description = "Go ahead and test the model! and don't forget to check the story behind it too!", colour = 0x000000)
                fields = [("Story", "Reply with 'story' if you want to learn more about the model before testing it", False),
                        ("Test", "Reply with 'test' in order to test the model yourself!", False),]
                for name, value, inline in fields:
                    embed.add_field(name = name, value = value, inline = inline)
                await message.channel.send(embed = embed)

                msg = await client.wait_for("message", check = check)
                sub_choice = msg.content.lower()

                if sub_choice == "story":
                    embed = discord.Embed(title = "S T O R Y", description = "Check out why we are using this model!", colour = 0x000000)
                    fields = [("Three Layer Neural Network Model", """
                    This Neural Network Model predicts whether you roasted your coffee beans poorly or perfectly based on two criteria. First, you choose the temperature you want to roast the beans at. Second, you choose the duration of roasting. Now, if you cook it at a low temperature, it doesn't get roasted and it ends up undercooked.
                    If you cook it, not for long enough, the duration is too short, it's also not a nicely roasted set of beans.
                    Finally, if you were to cook it either for too long or for on a high temperature, then you will end up with overcooked beans. 'Coffee Roasting at Home' suggests that the duration is best kept between 12 and 15 minutes while the temperature should be between 175 and 260 degrees Celsius.
                    So, what you want to do is to aim for the perfect combination of duration and temperature. And remember, as temperature rises, the duration should shrink.
                    """, False),
                              ("Test", "Reply with 'test' to back and test the model!", False)]
                    for name, value, inline in fields:
                        embed.add_field(name = name, value = value, inline = inline)
                    await message.channel.send(embed = embed)

                    msg = await client.wait_for("message", check = check)

                    if msg.content.lower() == "test":
                        await test4()
                
                elif sub_choice == "test":
                    await test4()
        
        if message.content == "-joke":
            joke_container = [
            #1
            """A Machine Learning algorithm walks into a bar.
The bartender says: "What will you have?"
The Algorithm says: "What's everyone else having?
            """,
            #2
            """What does a Machine Learning specialist and a Fashion designer have in common?
They both specialize in curve-fitting :p""",
            #3
            """Are you a Star Wars fan?
Cause may the FOREST be with you :')
            """,
            #4
            """Where did a Machine Learning Engineer go camping?
They went to a Random Forest :P
            """]
            await message.channel.send(random.choice(joke_container))

client = KLIYO()

TOKEN = os.getenv("TOKEN")

client.run(TOKEN)