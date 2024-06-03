using ActiveInference
using Distributed
using DataFrames
using Serialization
using Statistics
using Plots

#addprocs(10)

@everywhere begin
    include(raw"..\EnvsAndAgents\AIFInitFunctions.jl")
    include(raw"..\EnvsAndAgents\AlgoAgentsFunctions.jl")
    include(raw"..\EnvsAndAgents\GenerativeModel.jl")
    include(raw"..\EnvsAndAgents\PrisonersDilemmaEnv.jl")

    settings = Dict("use_param_info_gain" => false,
                    "use_states_info_gain" => false,
                    "action_selection" => "stochastic")

    parameters = Dict{String, Real}("lr_pB" => 0.1)

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

    function run_simulation(agent2_constructor, AlgoAgent_name, alpha, beta, lr_pB, fr_pB)
        # Initialize agents
        AIF_agent = init_AIF_agent_full(alpha, beta, lr_pB, fr_pB)
        AlgoAgent = agent2_constructor()

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

            q_pi_AIF, EFE_AIF = infer_policies!(AIF_agent)
            push!(AIF_EFE_store, EFE_AIF[1])
            
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
        
        return (AIF_agent="AIF_agent", AlgoAgent_name=AlgoAgent_name, score_AIF_agent=score_AIF_agent, score_AlgoAgent=score_AlgoAgent, alpha = alpha, beta = beta, lr_pB = lr_pB, fr_pB = fr_pB), (AIF_EFE_store)
    end
end

alphas = 1.0:2.0:32
betas = 0.5:0.5:5.5
lr_pBs = 0.1:0.1:1.0
fr_pBs = 0.1:0.1:1.0

length(alphas)*length(betas)*length(lr_pBs)*length(fr_pBs)*length(AlgoAgent_names)

# List of AlgoAgent constructors and names
AlgoAgent_constructors = [
    TitForTatAgent,
    TitFor2TatsAgent,
    TwoTitsFor1TatAgent,
    NastyForgivingTFTAgent,
    PavlovianAgent,
    GrofmanAgent,
    GrimTriggerAgent
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

# Create parameter combinations
param_combinations = [(alpha, beta, lr_pB, fr_pB, constructor, name) for alpha in alphas, beta in betas, lr_pB in lr_pBs, fr_pB in fr_pBs, (constructor, name) in zip(AlgoAgent_constructors, AlgoAgent_names)]

@time begin
    results = pmap(param -> begin
        (alpha, beta, lr_pB, fr_pB, constructor, name) = param
        run_simulation(constructor, name, alpha, beta, lr_pB, fr_pB)
    end, param_combinations)
end

# Separate the results
results_data = [res[1] for res in results]
EFE_AIF_CC_data = [res[2] for res in results]

results_df = DataFrame(results_data)

grouped_df = combine(groupby(results_df, [:alpha, :beta, :lr_pB, :fr_pB]),
                     :score_AIF_agent => mean => :mean_score_AIF,
                     :score_AlgoAgent => mean => :mean_score_Algo)


sorted_df = sort(grouped_df, :mean_score_AIF, rev=true)

EFE_AIF_CC_flat = [Dict("EFE_AIF" => efe, "param" => param) for (param, efe_list) in zip(param_combinations, EFE_AIF_CC_data) for efe in efe_list]
EFE_AIF_CC_df = DataFrame(EFE_AIF_CC_flat)

jls_filename = raw"SoCultRepo\SoCultExam\ResultData\EFE_results_df.jls"

open(jls_filename, "w") do io
    serialize(io, EFE_AIF_CC_df)
end

