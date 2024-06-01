using ActiveInference
using Distributed
using DataFrames
using Serialization
using Statistics
using Plots

#addprocs(10)

@everywhere begin
    include("../EnvsAndAgents/AIFInitFunctions.jl")
    include("../EnvsAndAgents/AlgoAgentsFunctions.jl")
    include("../EnvsAndAgents/GenerativeModel.jl")
    include("../EnvsAndAgents/PrisonersDilemmaEnv.jl")

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

    function run_simulation(AlgoAgent_constructor, AlgoAgent_name, lr_pB)
        # Initialize agents
        AIF_agent = init_AIF_agent_lr(lr_pB)
        AlgoAgent = AlgoAgent_constructor()

        # Initialize environment
        env = PrisonersDilemmaEnv()

        N_TRIALS = 1000

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
        
            # Store EFE (G)
            EFE = get_history(AIF_agent)["expected_free_energies"][end][1]
            push!(AIF_EFE_store, EFE)
        end
        
        return (AIF_agent="AIF_agent", AlgoAgent_name=AlgoAgent_name, score_AIF_agent=score_AIF_agent, score_AlgoAgent=score_AlgoAgent, lr_pB = lr_pB, EFE=AIF_EFE_store)
    end
end

lr_pBs = [0.1, 1.0]

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

# Use a tuple to collect both results_C and EFE_results_C
results_C = []
EFE_results_C = []

@time begin
    distributed_results_C = @distributed (vcat) for lr_pB in lr_pBs
        local_results_C = []
        local_EFE_results_C = []

        for (AlgoAgent_constructor, AlgoAgent_name) in zip(AlgoAgent_constructors, AlgoAgent_names)
            result = run_simulation(AlgoAgent_constructor, AlgoAgent_name, lr_pB)
            push!(local_results_C, result)
            for EFE in result[:EFE]
                push!(local_EFE_results_C, (lr_pB=lr_pB, AlgoAgent_name=AlgoAgent_name, EFE=EFE))
            end
        end
        [(local_results_C, local_EFE_results_C)]
    end

    # Aggregate the results_C
    for res in distributed_results_C
        local_results_C, local_EFE_results_C = res
        append!(results_C, local_results_C)
        append!(EFE_results_C, local_EFE_results_C)
    end
end

results_C_df = DataFrame(results_C)
EFE_results_C_df = DataFrame(EFE_results_C)

time_steps = 1:1000

average_EFE_C_df = DataFrame()

for lr_pB in unique(EFE_results_C_df.lr_pB)
    for t in time_steps
        EFE_values = []
        for AlgoAgent_name in unique(EFE_results_C_df.AlgoAgent_name)
            EFE_values = vcat(EFE_values, EFE_results_C_df[EFE_results_C_df.lr_pB .== lr_pB .&& EFE_results_C_df.AlgoAgent_name .== AlgoAgent_name, :EFE][t])
        end
        push!(average_EFE_C_df, (lr_pB=lr_pB, time_step=t, average_EFE=mean(EFE_values)))
    end
end

plot_data_C = DataFrame(lr_pB=Float64[], time_step=Int64[], average_EFE=Float64[])
for lr_pB in unique(average_EFE_C_df.lr_pB)
    for t in time_steps
        avg_EFE = mean(average_EFE_C_df[average_EFE_C_df.lr_pB .== lr_pB .&& average_EFE_C_df.time_step .== t, :average_EFE])
        push!(plot_data_C, (lr_pB=lr_pB, time_step=t, average_EFE=avg_EFE))
    end
end

plot_data_C

plot_data_C.average_EFE = (plot_data_C.average_EFE .* (-1))

plot_data_C

res = 10.0:10:1000

filtered_plot_C = filter(row -> row.time_step in res, plot_data_C)

plt = plot(title="Average EFE of cooperate for Different lr_pB",
           xlabel="Time Step",
           ylabel="Average EFE")

for lr_pB in unique(filtered_plot_C.lr_pB)
    lr_pB_data = filtered_plot_C[filtered_plot_C.lr_pB .== lr_pB, :]
    plot!(plt, lr_pB_data.time_step, lr_pB_data.average_EFE, label="lr_pB = $lr_pB")
end
theme(:juno)

display(plt)