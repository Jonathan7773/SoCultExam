using Pkg
using Revise
using Plots

#Pkg.activate(url = "C:\\Users\\Samuel\\Desktop\\Julia_projects\\Julia-Development")
Pkg.develop(path=raw"C:\Users\jonat\Desktop\University\Exam\4_Semester\Continued_ActiveInference\Dev_Branch\ActiveInference.jl")
#Pkg.develop(path="C:\\Users\\Samuel\\dev\\ActiveInference")

#Pkg.add(url="https://github.com/samuelnehrer02/ActiveInference.jl.git")

using ActiveInference
using LinearAlgebra

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
β = 1
C[1] = softmax(C[1] * β)

pB = deepcopy(B_matrix)


####= I am not sure how to set this up! xD 
for i in eachindex(pB)
    pB[i] = pB[i] .* 2.0
end



settings=Dict("use_param_info_gain" => false,
              "use_states_info_gain" => false,
              "action_selection" => "deterministic")

parameters_samuel=Dict{String, Real}("lr_pB" => 0.6)
parameters_jonathan=Dict{String, Real}("lr_pB" => 0.6)


SAMUEL = init_aif(A_matrix, B_matrix;
                    C=C,
                    pB=pB,
                    settings=settings,
                    parameters=parameters_samuel);

JONATHAN = init_aif(A_matrix, B_matrix;
                    C=C,
                    pB=pB,
                    settings=settings,
                    parameters=parameters_jonathan);


env = PrisonersDilemmaEnv()

N_TRIALS = 160

# Starting Observation
obs1 = [1]
obs2 = [1]

actions_samuel_store = []
actions_jonathan_store = []

obs_samuel_store = []
obs_jonathan_store = []

@time begin
    for t in 1:N_TRIALS

        infer_states!(SAMUEL, obs1)
        infer_states!(JONATHAN, obs2)


        # Update Transitions SAMUEL
        if get_states(SAMUEL)["action"] !== missing
            QS_prev = get_history(SAMUEL)["posterior_states"][end-1]
            update_B!(SAMUEL, QS_prev)
        end

        # Update Transitions JONATHAN
        if get_states(JONATHAN)["action"] !== missing
            QS_prev = get_history(JONATHAN)["posterior_states"][end-1]
            update_B!(JONATHAN, QS_prev)
        end

        infer_policies!(SAMUEL)
        infer_policies!(JONATHAN)

        action_SAMUEL = sample_action!(SAMUEL)
        action_SAMUEL = Int(action_SAMUEL[1])
        push!(actions_samuel_store, action_SAMUEL)


        action_JONATHAN = sample_action!(JONATHAN)
        action_JONATHAN = Int(action_JONATHAN[1])
        push!(actions_jonathan_store, action_JONATHAN)

        println(" **** At Trial: $(t) **** \n Samuel Plays: $(action_SAMUEL) \n Jonathan Plays: $(action_JONATHAN)")
        obs1, obs2, score_SAMUEL, score_JONATHAN = trial(env, action_SAMUEL, action_JONATHAN)
        obs1 = [findfirst(isequal(obs1), conditions)]
        obs2 = [findfirst(isequal(obs2), conditions)]

        push!(obs_samuel_store, obs1)
        push!(obs_jonathan_store, obs2)
        
        println("Samuel OBSERVES: $(obs1) \n SCORE SAMUEL: $(score_SAMUEL)")
        println("JONATHAN OBSERVES: $(obs2) \n SCORE JONATHAN: $(score_JONATHAN)")

    end 
end
action_matrix = [actions_samuel_store'; actions_jonathan_store']

cmap = [:springgreen, :brown2]

# Plot the heatmap
heatmap(action_matrix, color=cmap,
        clims = (1,2),
        legend=false, ylabel="Agent",
        yticks=(1:2, ["Samuel", "Jonathan"]),
        xticks=(0:10:160),
        size=(1000, 100))


#############  #############  #############  #############  ############# 
##########################  BIG SIMULATION ########################### 

using Random
learning_rates = 0.01:0.01:0.60
results = []

function init_agents(lr_samuel, lr_jonathan)
    parameters_samuel = Dict{String, Real}("lr_pB" => lr_samuel)
    parameters_jonathan = Dict{String, Real}("lr_pB" => lr_jonathan)

    SAMUEL = init_aif(A_matrix, B_matrix;
                      C=C,
                      pB=pB,
                      settings=settings,
                      parameters=parameters_samuel)

    JONATHAN = init_aif(A_matrix, B_matrix;
                        C=C,
                        pB=pB,
                        settings=settings,
                        parameters=parameters_jonathan)

    return SAMUEL, JONATHAN
end

@time begin
    # Loop over all combinations of learning rates
    for lr_samuel in learning_rates
        for lr_jonathan in learning_rates
            # Reinitialize agents
            SAMUEL, JONATHAN = init_agents(lr_samuel, lr_jonathan)
            
            # Initialize environment
            env = PrisonersDilemmaEnv()

            N_TRIALS = 160

            # Starting Observation
            obs1 = [1]
            obs2 = [1]

            actions_samuel_store = []
            actions_jonathan_store = []

            obs_samuel_store = []
            obs_jonathan_store = []

            score_SAMUEL = 0
            score_JONATHAN = 0

            # Run simulation
            for t in 1:N_TRIALS
                infer_states!(SAMUEL, obs1)
                infer_states!(JONATHAN, obs2)

                # Update Transitions SAMUEL
                if get_states(SAMUEL)["action"] !== missing
                    QS_prev = get_history(SAMUEL)["posterior_states"][end-1]
                    update_B!(SAMUEL, QS_prev)
                end

                # Update Transitions JONATHAN
                if get_states(JONATHAN)["action"] !== missing
                    QS_prev = get_history(JONATHAN)["posterior_states"][end-1]
                    update_B!(JONATHAN, QS_prev)
                end

                infer_policies!(SAMUEL)
                infer_policies!(JONATHAN)

                action_SAMUEL = sample_action!(SAMUEL)
                action_SAMUEL = Int(action_SAMUEL[1])
                push!(actions_samuel_store, action_SAMUEL)

                action_JONATHAN = sample_action!(JONATHAN)
                action_JONATHAN = Int(action_JONATHAN[1])
                push!(actions_jonathan_store, action_JONATHAN)

                obs1, obs2, score_SAMUEL, score_JONATHAN = trial(env, action_SAMUEL, action_JONATHAN)
                obs1 = [findfirst(isequal(obs1), conditions)]
                obs2 = [findfirst(isequal(obs2), conditions)]

                push!(obs_samuel_store, obs1)
                push!(obs_jonathan_store, obs2)

            end

            # Save results 
            push!(results, (lr_samuel=lr_samuel, lr_jonathan=lr_jonathan, score_SAMUEL=score_SAMUEL, score_JONATHAN=score_JONATHAN))
        end
    end
end

using DataFrames
using Serialization
using Plots

#results_df = deserialize("data_symmetrical_deterministic.jls")

results_df = DataFrame(results)
results_df[!, :total_reward] = results_df[!, :score_SAMUEL] .+ results_df[!, :score_JONATHAN]

pivot_df = unstack(results_df, :lr_jonathan, :lr_samuel, :total_reward)

x = unique(results_df[!, :lr_samuel])
y = unique(results_df[!, :lr_jonathan])
z = Matrix(pivot_df[:, Not(:lr_jonathan)])

heatmap(x, y, z, xlabel="Agent 1 Learning Rates", ylabel="Agent 2 Learning Rates", 
        title="Accumulated Total Reward", colorbar_title="Total Reward",
        color=:haline, size =(800,700))


#=
surface(x, y, z, title="Accumulated Total Reward", color=:haline, camera = (-40,60))
=#




