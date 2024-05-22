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

# Init Tester Agent

Agent_T = TesterAgent()

# Initialize environment
env = PrisonersDilemmaEnv()

N_TRIALS = 10

# Starting Observation
obs1 = [1]
obs2 = [1]

actions_AIF_store = []
actions_Tester_store = []

obs_AIF_store = []
obs_Tester_store = []

score_AIF_agent = 0
score_Tester_agent = 0


# Run simulation
for t in 1:N_TRIALS

    infer_states!(AIF_agent, obs1)

    # Taking the first observation as last action
    update_AlgoAgent(Agent_T, obs2)

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

    action_Tester_agent = choose_action_AlgoAgent(Agent_T)
    action_Tester_agent = Int(action_Tester_agent[1])
    push!(actions_Tester_store, action_Tester_agent)

    println(" **** At Trial: $(t) **** \n AIF_agent Plays: $(action_AIF_agent) \n Tester_agent Plays: $(action_Tester_agent)")
    obs1, obs2, score_AIF_agent, score_Tester_agent = trial(env, action_AIF_agent, action_Tester_agent)
    obs1 = [findfirst(isequal(obs1), conditions)]
    obs2 = [findfirst(isequal(obs2), conditions)]

    push!(obs_AIF_store, obs1)
    push!(obs_Tester_store, obs2)
    
    println("AIF_agent OBSERVES: $(obs1) \n SCORE AIF_agent: $(score_AIF_agent)")
    println("Tester_agent OBSERVES: $(obs2) \n SCORE Tester_agent: $(score_Tester_agent)")

end

action_matrix = [actions_AIF_store'; actions_Tester_store']

cmap = ["green", "red"]

# Plot the heatmap
heatmap(action_matrix, color=cmap,
        clims = (1,2),
        legend=false, ylabel="Agent",
        yticks=(1:2, ["AIF", "TesterAgent"]),
        size=(1000, 80))







