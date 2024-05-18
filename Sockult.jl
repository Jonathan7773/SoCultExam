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

B_matrix[1]

C = array_of_any_zeros(4)
C[1][1] = 2.0 # CC
C[1][2] = 0.0 # CD
C[1][3] = 3.0 # DC
C[1][4] = 1.0 # DD

# Parameterize and keep rations
beta = 10.0
C[1] = softmax(C[1] * beta)
C[1]
pB = deepcopy(B_matrix)

for i in eachindex(pB)
    pB[i] = pB[i] .* 2.0
end

settings=Dict("use_param_info_gain" => true,
              "action_selection" => "stochastic")

parameters=Dict("alpha" => 16.0,
                "lr_pB" => 0.1)


SAMUEL = init_aif(A_matrix, B_matrix;
                    C=C,
                    pB=pB,
                    settings=settings,
                    parameters=parameters);

JONATHAN = init_aif(A_matrix, B_matrix;
                    C=C,
                    pB=pB,
                    settings=settings,
                    parameters=parameters);


mutable struct PrisonersDilemmaEnv
    score_agent_1::Int
    score_agent_2::Int
    conditions::Vector{String}

    function PrisonersDilemmaEnv()
        new(0, 0, ["CC", "CD", "DC", "DD"])
    end
end

function trial(env::PrisonersDilemmaEnv, action_1::Int, action_2::Int)

    if action_1 == 1 && action_2 == 1
        obs_1 = 1  # CC for Agent 1
        obs_2 = 1  # CC for Agent 2
        points_1, points_2 = 3, 3  
    elseif action_1 == 1 && action_2 == 2
        obs_1 = 2  # CD for Agent 1
        obs_2 = 3  # DC for Agent 2
        points_1, points_2 = 0, 5  
    elseif action_1 == 2 && action_2 == 1
        obs_1 = 3  # DC for Agent 1
        obs_2 = 2  # CD for Agent 2
        points_1, points_2 = 5, 0  
    elseif action_1 == 2 && action_2 == 2
        obs_1 = 4  # DD for Agent 1
        obs_2 = 4  # DD for Agent 2
        points_1, points_2 = 1, 1 
    end

    env.score_agent_1 += points_1
    env.score_agent_2 += points_2

    return env.conditions[obs_1], env.conditions[obs_2], env.score_agent_1, env.score_agent_2
end

env = PrisonersDilemmaEnv()

N_TRIALS = 160

# Starting Observation
obs1 = [1]
obs2 = [1]

actions_samuel_store = []
actions_jonathan_store = []

obs_samuel_store = []
obs_jonathan_store = []

for t in 1:N_TRIALS

    infer_states!(SAMUEL, obs1)
    infer_states!(JONATHAN, obs2)


    # Update Transitions SAMUEL
    if get_states(SAMUEL)["action"] !== missing
        QS_prev = get_history(SAMUEL)["posterior_states"][end-1]
        #SAMUEL.qs_current = process_observation(obs1, length(A_matrix), observations)
        update_B!(SAMUEL, QS_prev)
    end

    # Update Transitions JONATHAN
    if get_states(JONATHAN)["action"] !== missing
        QS_prev = get_history(JONATHAN)["posterior_states"][end-1]
        #JONATHAN.qs_current = process_observation(obs2, length(A_matrix), observations)
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

actions_samuel_store[160]

action_matrix = [actions_samuel_store'; actions_jonathan_store']

cmap = ["green", "red"]

# Plot the heatmap
heatmap(action_matrix, color=cmap,
        clims = (1,2),
        legend=false, ylabel="Agent",
        yticks=(1:2, ["Samuel", "Jonathan"]),
        size=(1000, 80))

JONATHAN
SAMUEL.B[1]



likelihood = [0, -0.37, -0.37, -0.37]
qs = [0.25, 0.25, 0.25, 0.25]

spm_dot(likelihood, qs)
dot(likelihood, qs)

Env

Env._reward_condition_idx = 2
Env.reward_condition = [0, 1]





-1e2

-0.0000001











process_observation(obs1, length(A_matrix), observations)

JONATHAN.pB[1]
SAMUEL.pB[1]

get_history(JONATHAN)["action"]
get_history(SAMUEL)["action"]

SAMUEL.action
actions_jonathan_store


Int(sample_action!(SAMUEL)[1])

obs_jonathan_store
obs_samuel_store

obs_t = [4]

obs1
infer_states!(SAMUEL, obs1)

SAMUEL.A[1]
obs_samuel_sto
qs_current_t = SAMUEL.qs_current

kk = update_posterior_states(SAMUEL.A, obs_samuel_store[2], prior = SAMUEL.prior)
kk
observations

obs1_pro = process_observation(obs1, length(A_matrix), observations)

SAMUEL.prior


fixed_point_iteration1(SAMUEL.A, obs1_pro, observations, states, prior = SAMUEL.prior)


n_modalities = length(observations)
n_factors = length(states)

likelihood = get_joint_likelihood(SAMUEL.A, obs1_pro, states)
likelihood = spm_log_single(likelihood)

qs = Array{Any}(undef, n_factors)

for factor in 1:n_factors
    qs[factor] = ones(states[factor]) / states[factor]
end

prior = spm_log_array_any(SAMUEL.prior) 

prev_vfe = calc_free_energy(qs, prior, length(B_matrix))

qL = dot_likelihood(likelihood, qs[1])
qL = spm_dot(likelihood, qs[1])

to_array_of_any(softmax(qL .+ prior[1]))

qL
likelihood

function fixed_point_iteration1(A::Vector{Any}, obs::Vector{Any}, num_obs::Vector{Int64}, num_states::Vector{Int64}; prior::Union{Nothing, Vector{Any}}=nothing, num_iter=10, dF=1.0, dF_tol=0.001)
    n_modalities = length(num_obs)
    n_factors = length(num_states)

    # Get joint likelihood
    likelihood = get_joint_likelihood(A, obs, num_states)
    likelihood = spm_log_single(likelihood)

    # Initialize posterior and prior
    qs = Array{Any}(undef, n_factors)
    for factor in 1:n_factors
        qs[factor] = ones(num_states[factor]) / num_states[factor]
    end

    if prior === nothing
        prior = array_of_any_uniform(num_states)
    end
    
    prior = spm_log_array_any(prior) 

    # Initialize free energy
    prev_vfe = calc_free_energy(qs, prior, n_factors)

    # Single factor condition
    if n_factors == 1
        qL = spm_dot(likelihood, qs[1])  
        return to_array_of_any(softmax(qL .+ prior[1]))
    else
        # Run Iteration 
        curr_iter = 0
        while curr_iter < num_iter && dF >= dF_tol
            qs_all = qs[1]
            for factor in 2:n_factors
                qs_all = qs_all .* reshape(qs[factor], tuple(ones(Int, factor - 1)..., :, 1))
            end
            LL_tensor = likelihood .* qs_all

            for factor in 1:n_factors
                qL = zeros(size(qs[factor]))
                for i in 1:size(qs[factor], 1)
                    qL[i] = sum([LL_tensor[indices...] / qs[factor][i] for indices in Iterators.product([1:size(LL_tensor, dim) for dim in 1:n_factors]...) if indices[factor] == i])
                end
                qs[factor] = softmax(qL + prior[factor])
            end

            # Recompute free energy
            vfe = calc_free_energy(qs, prior, n_factors, likelihood)

            # Update stopping condition
            dF = abs(prev_vfe - vfe)
            prev_vfe = vfe

            curr_iter += 1
        end

        return qs
    end
end



































SAMUEL.qs_current




prior_t = [1.0666666666666647e-15, 1.0666666666666647e-15, 0.4999999999999989, 0.4999999999999989]


SAMUEL.action = [2]
SAMUEL.qs_current
SAMUEL.prior = [0.25, 0.25, 0.25, 0.25]

SAMUEL.action = [2]




get



