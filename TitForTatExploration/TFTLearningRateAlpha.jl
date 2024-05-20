using Pkg
using Revise

#====================================== ***PATHS*** ======================================# 
#Pkg.activate("C:\\Users\\Samuel\\Desktop\\Julia_projects\\Julia-Development")
#Pkg.develop(path="C:\\Users\\Samuel\\dev\\ActiveInference")

using LinearAlgebra
using ActiveInference
using Plots
using StatsPlots
using Distributions
using ActionModels

###### Defining environment ###########

include(raw"..\EnvsAndAgents\PrisonersDilemmaEnv.jl")
include(raw"..\EnvsAndAgents\TitForTatAgent.jl")

env = PrisonersDilemmaEnv()

TFT_agent = TitForTatAgent()

########## Creating Agents ##################

conditions = ["CC","CD","DC", "DD"]

states = [4]
observations = [4]
actions = ["COOPERATE", "DEFECT"]

controls = [length(actions)]

A_matrix, B_matrix = generate_random_GM(states, observations, controls)

A_matrix[1] = Matrix{Float64}(I, 4, 4)


# When you cooperate, the is a zero probability that the observation will start with D and VV
for i in eachindex(B_matrix)
    B_matrix[i] .= 0.0
end

# other conditions are 50/50
B_matrix[1][1,:,1] = [0.5,0.5,0.5,0.5]
B_matrix[1][2,:,1] = [0.5,0.5,0.5,0.5]

B_matrix[1][3,:,2] = [0.5,0.5,0.5,0.5]
B_matrix[1][4,:,2] = [0.5,0.5,0.5,0.5]


C = array_of_any_zeros(4)
C[1][1] = 3.0 # CC
C[1][2] = 1.0 # CD
C[1][3] = 4.0 # DC
C[1][4] = 2.0 # DD

# Parameterize preferences
β = 2
C[1] = softmax(C[1] * β)

pB = deepcopy(B_matrix)
pB[1]

####= I am not sure how to set this up! xD 
for i in eachindex(pB)
    pB[i] = pB[i] .* 2.0
end


settings=Dict("use_param_info_gain" => false,
              "use_states_info_gain" => false,
              "action_selection" => "deterministic")


learning_rates = 0.01:0.01:1.0
alphas = 1.0:1.0:32.0

results = []

total_iterations = length(learning_rates) * length(alphas)
current_iteration = 0

@time begin
    # Loop over all combinations of learning rates
    for lr_AIF in learning_rates
        for alpha_AIF in alphas

            current_iteration += 1
            println("Progress: $current_iteration / $total_iterations")

            # Reinitialize agents
            AIF_agent = init_agent_lr_alpha(lr_AIF, alpha_AIF)
            
            # Initialize environment
            env = PrisonersDilemmaEnv()

            N_TRIALS = 500

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
                    #AIF_agent.qs_current = process_observation(obs1, length(A_matrix), observations)
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
    
            # Save results 
            push!(results, (lr_AIF=lr_AIF, alpha_AIF, score_AIF_agent=score_AIF_agent, score_TFT_agent=score_TFT_agent))
        end
    end
end

results_df = DataFrame(results)

results_df[!, :total_reward] = results_df[!, :score_AIF_agent] .+ results_df[!, :score_TFT_agent]

x = unique(results_df[!, :lr_AIF])
y = unique(results_df[!, :alpha_AIF])

pivot_df = unstack(results_df, :lr_AIF, :alpha_AIF, :total_reward)
z = Matrix(pivot_df[:, Not(:lr_AIF)])
theme(:juno)

heatmap(y, x, z,
        xlabel="Alphas", ylabel="Learning Rates",
        title="Accumulated Total Reward", colorbar_title="Total Reward", size=(800, 700),
        color=:inferno)


# z_lr = (results_df[results_df[!, :alpha_AIF] .== 1, :])[!, :total_reward]
# plot(x, z_lr)



