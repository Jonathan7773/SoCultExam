using Pkg
using Optim
using Distributions
using Random
using Statistics
using ActiveInference
using Distributed
using ForwardDiff


include(raw"..\..\EnvsAndAgents\AIFInitFunctions.jl")
include(raw"..\..\EnvsAndAgents\AlgoAgentsFunctions.jl")
include(raw"..\..\EnvsAndAgents\GenerativeModel.jl")
include(raw"..\..\EnvsAndAgents\PrisonersDilemmaEnv.jl")

settings = Dict("use_param_info_gain" => false,
                "use_states_info_gain" => false,
                "action_selection" => "deterministic")

env = PrisonersDilemmaEnv()

# Initialize all agents
AlgoAgent_constructors = [
    TitForTatAgent,
    TitFor2TatsAgent,
    TwoTitsFor1TatAgent,
    NastyForgivingTFTAgent,
    PavlovianAgent,
    GrofmanAgent,
    GrimTriggerAgent,
    ()
]

AlgoAgent_names = [
    "TitForTatAgent",
    "TitFor2TatsAgent",
    "TwoTitsFor1TatAgent",
    "NastyForgivingTFTAgent",
    "PavlovianAgent",
    "GrofmanAgent",
    "GrimTriggerAgent"
]

function run_simulation(AlgoAgent_constructor, AlgoAgent_name, alpha, beta, lr_pB, fr_pB)
    # Initialize agents
    AIF_agent = init_AIF_agent_full(alpha, beta, lr_pB, fr_pB)
    AlgoAgent = AlgoAgent_constructor()

    # Initialize environment
    env = PrisonersDilemmaEnv()

    N_TRIALS = 10

    # Starting Observation
    obs1 = [1]
    obs2 = [1]

    actions_AIF_agent_store = []
    actions_AlgoAgent_store = []

    obs_AIF_agent_store = []
    obs_AlgoAgent_store = []

    AIF_EFE_store = []

    score_AIF_agent = 0
    score_AlgoAgent = 0

    # Run simulation
    for t in 1:N_TRIALS

        infer_states!(AIF_agent, obs1)
        update_AlgoAgent(AlgoAgent, obs2)

        if get_states(AIF_agent)["action"] !== missing
            QS_prev = get_history(AIF_agent)["posterior_states"][end-1]
            update_B!(AIF_agent, QS_prev)
        end

        infer_policies!(AIF_agent)
        
        action_AIF_agent = sample_action!(AIF_agent)
        action_AlgoAgent = choose_action_AlgoAgent(AlgoAgent)
        
        action_AIF_agent = Int(action_AIF_agent[1])
        push!(actions_AIF_agent_store, action_AIF_agent)
    
        action_AlgoAgent = Int(action_AlgoAgent[1])
        push!(actions_AlgoAgent_store, action_AlgoAgent)
    
        obs1, obs2, score_AIF_agent, score_AlgoAgent = trial(env, action_AIF_agent, action_AlgoAgent)
        obs1 = [findfirst(isequal(obs1), conditions)]
        obs2 = [findfirst(isequal(obs2), conditions)]
    
        push!(obs_AIF_agent_store, obs1)
        push!(obs_AlgoAgent_store, obs2)
    
    end
    
    return score_AIF_agent
end

function objective_function(params)
    alpha, beta, lr_pB, fr_pB = params
    total_score = 0

    # Iterate over all AlgoAgents
    for (AlgoAgent_constructor, AlgoAgent_name) in zip(AlgoAgent_constructors, AlgoAgent_names)
        score_AIF_agent = run_simulation(AlgoAgent_constructor, AlgoAgent_name, alpha, beta, lr_pB, fr_pB)
        total_score += score_AIF_agent
    end

    return -total_score  # We minimize the negative total score to maximize the actual total score
end

# Initial parameter values
initial_params = [1.0, 2.5, 0.5, 0.5]

# Wrap the objective function
obj_fun = TwiceDifferentiable(objective_function, initial_params; autodiff = :forward)

result = optimize(obj_fun, initial_params, Newton())

