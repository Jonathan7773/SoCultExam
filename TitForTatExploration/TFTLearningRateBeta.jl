using Pkg
using Revise

#====================================== ***PATHS*** ======================================# 
#Pkg.develop(path=raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\Continued_ActiveInference\Dev_Branch\ActiveInference.jl")

using LinearAlgebra
using ActiveInference
using Plots
using StatsPlots
using Distributions
using ActionModels
using DataFrames

###### Defining environment ###########

include(raw"..\EnvsAndAgents\AIFInitFunctions.jl")
include(raw"..\EnvsAndAgents\AlgoAgentsFunctions.jl")
include(raw"..\EnvsAndAgents\GenerativeModel.jl")
include(raw"..\EnvsAndAgents\PrisonersDilemmaEnv.jl")

env = PrisonersDilemmaEnv()

TFT_agent = TitForTatAgent()

########## Creating Agents ##################

settings=Dict("use_param_info_gain" => false,
              "use_states_info_gain" => false,
              "action_selection" => "deterministic")


########## Creating TitForTatAgent ##############
learning_rates = 0.1:0.1:1.0
betas = 0.5:0.5:5.0

results = []

total_iterations = length(learning_rates) * length(betas)
current_iteration = 0

@time begin
    # Loop over all combinations of learning rates
    for lr_AIF in learning_rates
        for beta_AIF in betas

            current_iteration += 1
            println("Progress: $current_iteration / $total_iterations")

            # Reinitialize agents
            AIF_agent = init_AIF_agent_beta_lr_pB(beta_AIF, lr_AIF)
            
            # Initialize environment
            env = PrisonersDilemmaEnv()

            N_TRIALS = 10

            # Starting Observation
            obs1 = [1]
            obs2 = [1]

            actions_AIF_store = []
            actions_TFT_store = []

            obs_AIF_store = []
            obs_TFT_store = []

            score_AIF_agent = 0
            score_TFT_agent = 0

            # Run simulation
            for t in 1:N_TRIALS

                infer_states!(AIF_agent, obs1)
            
                # Taking the first observation as last action
                update_AlgoAgent(TFT_agent, obs2)
            
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
            
                action_TFT_agent = choose_action_AlgoAgent(TFT_agent)
                action_TFT_agent = Int(action_TFT_agent[1])
                push!(actions_TFT_store, action_TFT_agent)
            
                obs1, obs2, score_AIF_agent, score_TFT_agent = trial(env, action_AIF_agent, action_TFT_agent)
                obs1 = [findfirst(isequal(obs1), conditions)]
                obs2 = [findfirst(isequal(obs2), conditions)]
            
                push!(obs_AIF_store, obs1)
                push!(obs_TFT_store, obs2)
            
            end
    
            # Save results 
            push!(results, (lr_AIF=lr_AIF, beta_AIF = beta_AIF, score_AIF_agent=score_AIF_agent, score_TFT_agent=score_TFT_agent))
        end
    end
end

results_df = DataFrame(results)

results_df[!, :total_reward] = results_df[!, :score_AIF_agent] .+ results_df[!, :score_TFT_agent]

x = unique(results_df[!, :lr_AIF])
y = unique(results_df[!, :beta_AIF])

pivot_df = unstack(results_df, :beta_AIF, :lr_AIF, :score_AIF_agent)
z = Matrix(pivot_df[:, Not(:beta_AIF)])

heatmap(x, y, z,
        xlabel="Learning Rates", ylabel="Betas",
        title="Accumulated Total Reward", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)


z_lr = (results_df[results_df[!, :beta_AIF] .== 1.0, :])[!, :total_reward]

plot(x, z_lr, xlabel = "Learning Rates", ylabel = "Total Score", title = "Betas for lr = 1.0", legend = false)


AIF_agent = init_AIF_agent_beta_lr_pB(0.5, 1.5)















############## Parallel Computing Test ##################
using Distributed

addprocs(10)

@everywhere begin

    include(raw"..\EnvsAndAgents\PrisonersDilemmaEnv.jl")
    include(raw"..\EnvsAndAgents\TitForTatAgent.jl")
    include(raw"..\EnvsAndAgents\GenerativeModel.jl")

    settings=Dict("use_param_info_gain" => false,
              "use_states_info_gain" => false,
              "action_selection" => "deterministic")

    env = PrisonersDilemmaEnv()

    TFT_agent = TitForTatAgent()

    function run_simulation(lr_AIF, beta_AIF)
        # Reinitialize agents
        AIF_agent = init_agent_lr_beta(lr_AIF, beta_AIF)
        
        # Initialize environment
        env = PrisonersDilemmaEnv()

        N_TRIALS = 100

        # Starting Observation
        obs1 = [1]
        obs2 = [1]

        actions_AIF_store = []
        actions_TFT_store = []

        obs_AIF_store = []
        obs_TFT_store = []

        score_AIF_agent = 0
        score_TFT_agent = 0

        # Run simulation
        for t in 1:N_TRIALS
            infer_states!(AIF_agent, obs1)
        
            # Taking the first observation as last action
            update_TFT(TFT_agent, obs2)
        
            # Update Transitions SAMUEL
            if get_states(AIF_agent)["action"] !== missing
                QS_prev = get_history(AIF_agent)["posterior_states"][end-1]
                update_B!(AIF_agent, QS_prev)
            end
        
            infer_policies!(AIF_agent)
        
            action_AIF_agent = sample_action!(AIF_agent)
            action_AIF_agent = Int(action_AIF_agent[1])
            push!(actions_AIF_store, action_AIF_agent)
        
            action_TFT_agent = choose_action_TFT(TFT_agent)
            action_TFT_agent = Int(action_TFT_agent[1])
            push!(actions_TFT_store, action_TFT_agent)
        
            obs1, obs2, score_AIF_agent, score_TFT_agent = trial(env, action_AIF_agent, action_TFT_agent)
            obs1 = [findfirst(isequal(obs1), conditions)]
            obs2 = [findfirst(isequal(obs2), conditions)]
        
            push!(obs_AIF_store, obs1)
            push!(obs_TFT_store, obs2)
        
        end
        
        return (lr_AIF=lr_AIF, beta_AIF=beta_AIF, score_AIF_agent=score_AIF_agent, score_TFT_agent=score_TFT_agent)
    end
end

learning_rates = 0.005:0.005:1.0
betas = 0.05:0.05:5.5

results = []

total_iterations = Int64((length(learning_rates) * length(betas))/length(procs()))
current_iteration = 0

@time begin
    results = @distributed (append!) for lr_AIF in learning_rates
        local_results = []
        for beta_AIF in betas
            global current_iteration += 1
            println("Progress: $current_iteration / $total_iterations")

            result = run_simulation(lr_AIF, beta_AIF)
            push!(local_results, result)
        end
        local_results
    end
end


results_df = DataFrame(results)

results_df[!, :total_reward] = results_df[!, :score_AIF_agent] .+ results_df[!, :score_TFT_agent]

x = unique(results_df[!, :lr_AIF])
y = unique(results_df[!, :beta_AIF])

pivot_df = unstack(results_df, :beta_AIF, :lr_AIF, :total_reward)
z = Matrix(pivot_df[:, Not(:beta_AIF)])
theme(:juno)

heatmap(x, y, z,
        xlabel="Learning Rates", ylabel="Betas",
        title="Accumulated Total Reward", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)

