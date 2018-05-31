function theta = randTheta(size1,size2)

theta = zeros(size1,size2);
epsilon_init = 0.15;
theta = rand(size1,size2) * 2 * epsilon_init - epsilon_init;

end