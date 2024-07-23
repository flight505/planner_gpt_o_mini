# pip install openai gradio
# export OPENAI_API_KEY=""

import openai
import os
import gradio as gr

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_plans(user_query, n=5):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Plan and respond to the user query."},
            {"role": "user", "content": user_query},
        ],
        n=n,
        temperature=0.7,
        max_tokens=500,
        stop=[""],
    )
    plans = [
        choice.message.content
        for choice in response.choices
        if choice.message.content.strip() != ""
    ]
    if not plans:
        plans = ["Plan A", "Plan B", "Plan C"]
    return plans


def compare_plans(plan1, plan2):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Choose the better plan."},
            {
                "role": "user",
                "content": f"Plan 1: {plan1}\n\nPlan 2: {plan2}\n\nWhich plan is better? Respond with either '1' or '2'.",
            },
        ],
        temperature=0.2,
        max_tokens=10,
    )
    return (
        response.choices[0].message.content.strip()
        if response.choices[0].message.content.strip() != ""
        else "1"
    )


def evaluate_plans(plans, user_query):
    winners = plans
    while len(winners) > 1:
        next_round = []
        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                winner = (
                    winners[i]
                    if compare_plans(winners[i], winners[i + 1]) == "1"
                    else winners[i + 1]
                )
            else:
                winner = winners[i]
            next_round.append(winner)
        winners = next_round
    return winners[0] if winners else "No best plan found"


def generate_response(best_plan, user_query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Respond to the user query based on the plan.",
            },
            {
                "role": "user",
                "content": f"User Query: {user_query}\n\nPlan: {best_plan}\n\nGenerate a detailed response.",
            },
        ],
        temperature=0.5,
        max_tokens=2000,
    )
    return response.choices[0].message.content


def improved_ai_output(user_query, num_plans=20):
    plans = generate_plans(user_query, n=num_plans)
    best_plan = evaluate_plans(plans, user_query)
    final_response = generate_response(best_plan, user_query)
    return {
        "user_query": user_query,
        "best_plan": best_plan,
        "final_response": final_response,
    }


def chat(query):
    result = improved_ai_output(query)
    return result["final_response"]


interface = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=2, placeholder="Ask me anything..."),
    outputs=gr.Textbox(),
    title="My Planner",
    description="Get a personalized plan as per your requirement!",
)

if __name__ == "__main__":
    interface.launch()
