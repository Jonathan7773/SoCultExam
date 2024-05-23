using ActiveInference
using Distributed
using DataFrames
using Serialization
using Statistics
using Plots

#addprocs(60)

@everywhere begin
    include(raw"..\EnvsAndAgents\AIFInitFunctions.jl")
    include(raw"..\EnvsAndAgents\AlgoAgentsFunctions.jl")
    include(raw"..\EnvsAndAgents\GenerativeModel.jl")
    include(raw"..\EnvsAndAgents\PrisonersDilemmaEnv.jl")

    settings = Dict("use_param_info_gain" => false,
                    "use_states_info_gain" => false,
                    "action_selection" => "deterministic")

    env = PrisonersDilemmaEnv()

    # Initialize all agents
    AlgoAgent_constructors = [
        TitForTatAgent,
        ()
    ]

    AlgoAgent_names = [
        "TitForTatAgent"
    ]

    function run_simulation(AlgoAgent_constructor, AlgoAgent_name, beta, lr_pB)
        # Initialize agents
        AIF_agent = init_AIF_agent_beta_lr_pB(beta, lr_pB)
        AlgoAgent = AlgoAgent_constructor()

        # Initialize environment
        env = PrisonersDilemmaEnv()

        N_TRIALS = 20

        # Starting Observation
        obs1 = [1]
        obs2 = [1]

        actions_AIF_agent_store = []
        actions_AlgoAgent_store = []

        obs_AIF_agent_store = []
        obs_AlgoAgent_store = []

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
        
        return (AIF_agent="AIF_agent", AlgoAgent_name=AlgoAgent_name, score_AIF_agent=score_AIF_agent, score_AlgoAgent=score_AlgoAgent, beta = beta, lr_pB = lr_pB)
    end
end

betas = 0.2:0.2:5.5
lr_pBs = 0.01:0.01:1.0

betas = 1.5
lr_pBs = 0.5
# List of AlgoAgent constructors and names
AlgoAgent_constructors = [
    TitForTatAgent
]

AlgoAgent_names = [
    "TitForTatAgent"
]

results = []

@time begin
    results = @distributed (append!) for beta in betas
        local_results = []
        println("Running simulation: beta=$beta")
        for lr_pB in lr_pBs
            for (AlgoAgent_constructor, AlgoAgent_name) in zip(AlgoAgent_constructors, AlgoAgent_names)
                result = run_simulation(AlgoAgent_constructor, AlgoAgent_name, beta, lr_pB)
                push!(local_results, result)
            end
        end
        local_results
    end
end

results_df = DataFrame(results)

grouped_df = combine(groupby(results_df, [:beta, :lr_pB]),
                     :score_AIF_agent => sum => :total_score_AIF,
                     :score_AlgoAgent => sum => :total_score_Algo)


sorted_df = sort(grouped_df, :total_score_AIF, rev=true)

heatmap_df = unstack(sorted_df, :lr_pB, :beta, :total_score_AIF)

heatmap_matrix = Matrix(heatmap_df[:, 2:end])

heatmap(betas, lr_pBs, heatmap_matrix, xlabel="lr_pB", ylabel="beta", c=:inferno, title="Heatmap of Total Score AIF Agent")

#jls_filename = raw"SoCultExamProject\ResultData\results_1000Trials_stochastic_df.jls"

#open(jls_filename, "w") do io
#    serialize(io, sorted_df)
#end

function run_simulation(AlgoAgent_constructor, AlgoAgent_name, beta, lr_pB)
    # Initialize agents
    AIF_agent = init_AIF_agent_beta_lr_pB(beta, lr_pB)
    AlgoAgent = AlgoAgent_constructor()

    # Initialize environment
    env = PrisonersDilemmaEnv()

    N_TRIALS = 20

    # Starting Observation
    obs1 = [1]
    obs2 = [1]

    actions_AIF_agent_store = []
    actions_AlgoAgent_store = []

    obs_AIF_agent_store = []
    obs_AlgoAgent_store = []

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
    
    return (AIF_agent="AIF_agent", AlgoAgent_name=AlgoAgent_name, score_AIF_agent=score_AIF_agent, score_AlgoAgent=score_AlgoAgent, beta = beta, lr_pB = lr_pB)
end

run_simulation(TitForTatAgent, "TitForTatAgent", 1.5, 0.5)

settings = Dict("use_param_info_gain" => false,
                    "use_states_info_gain" => false,
                    "action_selection" => "deterministic")

include(raw"..\EnvsAndAgents\AIFInitFunctions.jl")
include(raw"..\EnvsAndAgents\AlgoAgentsFunctions.jl")
include(raw"..\EnvsAndAgents\GenerativeModel.jl")
include(raw"..\EnvsAndAgents\PrisonersDilemmaEnv.jl")


obs1 = [1]
obs2 = [1]

env = PrisonersDilemmaEnv()

actions_AIF_agent_store = []
actions_TFT_agent_store = []


obs_AIF_agent_store = []
obs_TFT_agent_store = []

N_TRIALS = 20

AIF_agent = init_AIF_agent_beta_lr_pB(1.5, 0.5)
TFT_agent = TitForTatAgent()

for t in 1:N_TRIALS
    println("-------------------- Timestep: $t --------------------")
    infer_states!(AIF_agent, obs1)
    update_AlgoAgent(TFT_agent, obs2)

    if get_states(AIF_agent)["action"] !== missing
        QS_prev = get_history(AIF_agent)["posterior_states"][end-1]
        update_B!(AIF_agent, QS_prev)
    end

    infer_policies!(AIF_agent)
    
    action_AIF_agent = sample_action!(AIF_agent)
    action_TFT_agent = choose_action_AlgoAgent(TFT_agent)
    
    action_AIF_agent = Int(action_AIF_agent[1])
    push!(actions_AIF_agent_store, action_AIF_agent)

    action_TFT_agent = Int(action_TFT_agent[1])
    push!(actions_TFT_agent_store, action_TFT_agent)

    println("Action AIF_agent: $action_AIF_agent")
    println("Action TFT_agent: $action_TFT_agent")

    obs1, obs2, score_AIF_agent, score_TFT_agent = trial(env, action_AIF_agent, action_TFT_agent)
    obs1 = [findfirst(isequal(obs1), conditions)]
    obs2 = [findfirst(isequal(obs2), conditions)]

    println("AIF_agent score: $(score_AIF_agent)")
    println("TFT_agent score: $(score_TFT_agent)")

    push!(obs_AIF_agent_store, obs1)
    push!(obs_TFT_agent_store, obs2)

end



