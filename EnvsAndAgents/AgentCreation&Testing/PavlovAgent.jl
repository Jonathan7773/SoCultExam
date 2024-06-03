using Pkg
using Plots

include(raw"..\AIFInitFunctions.jl")
include(raw"..\AlgoAgentsFunctions.jl")
include(raw"..\GenerativeModel.jl")
include(raw"..\PrisonersDilemmaEnv.jl")

settings=Dict("use_param_info_gain" => false,
              "use_states_info_gain" => false,
              "action_selection" => "deterministic")

# Reinitialize agents
AIF_agent = init_agent_lr_alpha(1.0, 16)

env = PrisonersDilemmaEnv()

Agent_P = PavlovianAgent()

N_TRIALS = 100

# Starting Observation
obs1 = [1]
obs2 = [1]

actions_AIF_store = []
actions_Pavlovian_store = []

obs_AIF_store = []
obs_Pavlovian_store = []

score_AIF_agent = 0
score_Pavlovian_agent = 0


# Run simulation
for t in 1:N_TRIALS

    infer_states!(AIF_agent, obs1)

    # Taking the first observation as last action
    update_AlgoAgent(Agent_P, obs2)

    # Update Transitions SAMUEL
    if get_states(AIF_agent)["action"] !== missing
        QS_prev = get_history(AIF_agent)["posterior_states"][end-1]
        #AIF_agent.qs_current = process_observation(obs1, length(A_matrix), observations)
        update_B!(AIF_agent, QS_prev)
    end

    infer_policies!(AIF_agent)

    action_AIF_agent = sample_action!(AIF_agent)
    action_AIF_agent = Int(action_AIF_agent[1])
    push!(actions_AIF_store, action_AIF_agent)

    action_Pavlovian_agent = choose_action_AlgoAgent(Agent_P)
    action_Pavlovian_agent = Int(action_Pavlovian_agent[1])
    push!(actions_Pavlovian_store, action_Pavlovian_agent)

    println(" **** At Trial: $(t) **** \n AIF_agent Plays: $(action_AIF_agent) \n Pavlovian_agent Plays: $(action_Pavlovian_agent)")
    obs1, obs2, score_AIF_agent, score_Pavlovian_agent = trial(env, action_AIF_agent, action_Pavlovian_agent)
    obs1 = [findfirst(isequal(obs1), conditions)]
    obs2 = [findfirst(isequal(obs2), conditions)]

    push!(obs_AIF_store, obs1)
    push!(obs_Pavlovian_store, obs2)
    
    println("AIF_agent OBSERVES: $(obs1) \n SCORE AIF_agent: $(score_AIF_agent)")
    println("Pavlovian_agent OBSERVES: $(obs2) \n SCORE Pavlovian_agent: $(score_Pavlovian_agent)")

end

action_matrix = [actions_AIF_store'; actions_Pavlovian_store']

cmap = ["green", "red"]

# Plot the heatmap
heatmap(action_matrix, color=cmap,
        clims = (1,2),
        legend=false, ylabel="Agent",
        yticks=(1:2, ["AIF", "PavlovianAgent"]),
        size=(1000, 80))





