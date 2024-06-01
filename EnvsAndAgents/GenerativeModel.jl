using LinearAlgebra

############## Basic generative model exclusive settings and params #################

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
β = 1.0
C[1] = softmax(C[1] * β)

pB = deepcopy(B_matrix)
pB[1]

####= I am not sure how to set this up! xD 
for i in eachindex(pB)
    pB[i] = pB[i] .* 0.1
end

E = array_of_any(1)

E[1] = softmax([2.0, 0.5])