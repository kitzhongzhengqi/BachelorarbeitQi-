%NSGA II Algorithm
input.init.saved_veh_file='PARALLEL_defaults_in';[error_code,resp]=adv_no_gui('initialize',input);
%初始化种群  
N_size=30, M_object=4, V_number=8,G_number=60;
minmin=[1 0 0 0.1 0 0.6 0.1 0 ];
maxmax=[30 20 10 1 1 0.9 0.6 10 ];
 mu=10;
mum=10;
 
for ii=1:N_size
    for jj=1:V_number
        parent_chromosome(ii,jj)=minmin(jj)+(maxmax(jj)-minmin(jj))*rand(1);
    end
end
 
 
% definite fitness funtion
fitness_function=[];   
 for i_function=1:G_number
     for j_function=1:1
         fitness_function(i_function,j_function)=0;
     end
 end
 
  fuel_function=[];  
 for i_fuel_function=1:G_number
     for j_fuel_function=1:1
          fuel_function(i_fuel_function,j_fuel_function)=0;
     end
 end 
 
 hc_function=[];  
 for i_hc_function=1:G_number
     for j_hc_function=1:1
          hc_function(i_hc_function,j_hc_function)=0;
     end
 end 
 
 co_function=[];  
 for i_co_function=1:G_number
     for j_co_function=1:1
          co_function(i_co_function,j_co_function)=0;
     end
 end 
 
 nox_function=[];  
 for i_nox_function=1:G_number
     for j_nox_function=1:1
          nox_function(i_nox_function,j_nox_function)=0;
     end
 end 
 
 
for i_generation=1:G_number % the number of generation
%simulated Binary Crossover (SBX)交叉
generationnumber=i_generation    % in order to see precess when running
p=1; q=1;
% Flags used to set if crossover and mutation were actually performed.
was_crossover = 0;
was_mutation = 0;
for i_cross=1:N_size/2
    if rand(1)<0.9
         % Initialize the children to be null vector.
        child_1 = [];
        child_2 = [];
        % Select the first parent
        parent_1 = round(N_size*rand(1));
        if parent_1 < 1
            parent_1 = 1;
        end
        % Select the second parent
        parent_2 = round(N_size*rand(1));
        if parent_2 < 1
            parent_2 = 1;
        end
        % Make sure both the parents are not the same.
        while isequal(parent_chromosome(parent_1,:),parent_chromosome(parent_2,:))
            parent_2 = round(N_size*rand(1));
            if parent_2 < 1
                parent_2 = 1;
            end
            break;
        end
        % Get the chromosome information for each randomnly selected parents
        parent_1=parent_chromosome(parent_1,:);
        parent_2=parent_chromosome(parent_2,:);
         for j_cross = 1 : V_number
            % SBX (Simulated Binary Crossover).
            % For more information about SBX refer the enclosed pdf file.
            % Generate a random number
            u(j_cross) = rand(1);
            if u(j_cross) <= 0.5
                bq(j_cross) = (2*u(j_cross))^(1/(mu+1));
            else
                bq(j_cross) = (1/(2*(1 - u(j_cross))))^(1/(mu+1));
            end
            % Generate the jth element of first child
            child_1(j_cross) = ...
                0.5*(((1 + bq(j_cross))*parent_1(j_cross)) + (1 - bq(j_cross))*parent_2(j_cross));
            % Generate the jth element of second child
            child_2(j_cross) = ...
                0.5*(((1 - bq(j_cross))*parent_1(j_cross)) + (1 + bq(j_cross))*parent_2(j_cross));
            % Make sure that the generated element is within the specified
            % decision space else set it to the appropriate extrema.
            if child_1(j_cross) > maxmax(j_cross)
                child_1(j_cross) = maxmax(j_cross);
            elseif child_1(j_cross) < minmin(j_cross)
                child_1(j_cross) = minmin(j_cross);
            end
            if child_2(j_cross) > maxmax(j_cross)
                child_2(j_cross) = maxmax(j_cross);
            elseif child_2(j_cross) < minmin(j_cross)
                child_2(j_cross) = minmin(j_cross);
            end
         end
      else
          child_1=parent_chromosome(i_cross,:);
          child_2=parent_chromosome(i_cross+1,:);
          
    end 
         child_crossover(p,:) = child_1;
         child_crossover(p+1,:) = child_2;
           p = p + 2;
end
    
  %变异
  % Mutation is based on polynomial mutation. 
for i_mu=1:N_size  
        
   if rand(1)<0.1
      % Select at random the parent.
        parent_3 = round(N_size*rand(1));
        if parent_3 < 1
            parent_3 = 1;
        end
        % Get the chromosome information for the randomnly selected parent.
        child_3 = child_crossover(parent_3,:);
        % Perform mutation on eact element of the selected parent.
        for j_mu = 1 : V_number
           r(j_mu) = rand(1);
           if r(j_mu) < 0.5
               delta(j_mu) = (2*r(j_mu))^(1/(mum+1)) - 1;
           else
               delta(j_mu) = 1 - (2*(1 - r(j_mu)))^(1/(mum+1));
           end
           % Generate the corresponding child element.
           child_3(j_mu) = child_3(j_mu) + delta(j_mu)*(maxmax(j_mu)-minmin(j_mu));
           % Make sure that the generated element is within the decision
           % space.
           if child_3(j_mu) > maxmax(j_mu)
               child_3(j_mu) = maxmax(j_mu);
           elseif child_3(j_mu) < minmin(j_mu)
               child_3(j_mu) = minmin(j_mu);
           end
        end
%         % Evaluate the objective function for the offspring and as before
%         % concatenate the offspring chromosome with objective value.
%        %     child_3(:,V + 1: M + V) = evaluate_objective(child_3, M, V);
%         % Set the mutation flag
%         was_mutation = 1;
%         was_crossover = 0;
   else  child_3=child_crossover(i_mu,:);
 
    end
    
    % Keep proper count and appropriately fill the child variable with all
    % the generated children for the particular generation.    
   
         child(q,:) = child_3;
         was_mutation = 0;
         q = q + 1;
          
end
 
% combine  parent_chromosome and child and the size is 2N
 
R_conbine=[parent_chromosome;child]
 
 [MMM nnn]=size(R_conbine)
 
 
 
 %Evaluate objective funtion
 clear nnn
 
  object=[];
 for iii=1:MMM
    for jjj=1:M_object      
       object(iii,jjj)=0;         
    end
 end
 
  
 performance=[];
 
 for iii=1:MMM
    for jjj=1:5      
       performance(iii,jjj)=0;         
    end
 end
 
 
 
 yyyy=0;
 
 fitness=[];   % definite fitness funtion
 for i_fitness=1:MMM
     for j_fitness=1:1
         fitness(i_fitness,j_fitness)=0;
     end
 end
 
 
  min_fuel=[];  
 for i_min_fuel=1:MMM
     for j_min_fuel=1:1
          min_fuel(i_min_fuel,j_min_fuel)=0;
     end
 end
 
 
  min_hc=[];  
 for i_min_hc=1:MMM
     for j_min_hc=1:1
         min_hc(i_min_hc,j_min_hc)=0;
     end
 end
 
 
  min_co=[];  
 for i_min_co=1:MMM
     for j_min_co=1:1
         min_co(i_min_co,j_min_co)=0;
     end
 end
     
     
 min_nox=[];   
 for i_min_nox=1:MMM
     for j_min_nox=1:1
         min_nox(i_min_nox,j_min_nox)=0;
     end
 end
     
        
     
 for i_pa=1:MMM
 
 aa=R_conbine(i_pa,1)
 bb=R_conbine(i_pa,2)
 cc=R_conbine(i_pa,3)
 dd=R_conbine(i_pa,4)
 ee=R_conbine(i_pa,5)
 ff=R_conbine(i_pa,6)
 gg=R_conbine(i_pa,7)
 hh=R_conbine(i_pa,8)
 
% %   input.modify.param={'fc_pwr_scale','mc_trq_scale','ess_cap_scale','cs_off_trq_frac','cs_min_trq_frac','cs_charge_trq','cs_electric_launch_spd_hi'};
% %   input.modify.value={aa,bb,cc,dd,ee,ff,gg};
% %    [a,b]=adv_no_gui('modify',input);
  input.modify.param={'cs_charge_trq','cs_electric_launch_spd_hi','cs_electric_launch_spd_lo','cs_min_trq_frac','cs_off_trq_frac','cs_hi_soc','cs_lo_soc','cs_electric_decel_spd'};
  input.modify.value={aa,bb,cc,dd,ee,ff,gg,hh};
  [error_code,resp]=adv_no_gui('modify',input) 
    
    
 
R1=10, R2=100;%definte penaty parameter
% %accel_test
input.accel.param={'spds'};input.accel.value={[0 96.5; 64.4 96.5;0 136.8]};[error_code,resp]=adv_no_gui('accel_test',input)
accel1=resp.accel.times(1);
accel2=resp.accel.times(2);
accel3=resp.accel.times(3);
input.grade.param={'grade','duration','speed'};input.grade.value={6.5,1200,88.5};[error_code,resp]=adv_no_gui('grade_test',input)    
grade_test=resp.grade.grade(1)   
  
input.cycle.param={'name','soc','socmenu','number'};input. cycle.value={'CYC_UDDS','on','zerodelta',1};[error_code,resp]=adv_no_gui('drive_cycle',input)
    
   fuel_test=resp.cycle.mpgge
   hc_test=resp.cycle.hc_gpm
   co_test=resp.cycle.co_gpm
   nox_test=resp.cycle.nox_gpm
   delta_soc=resp.cycle.delta_soc
 
%Constrain handle  definite R1=100, R2=10
t1=12-accel1
t2=5.3-accel2
t3=23.4-accel3
grade_value=grade_test-5
 
 
fuel=378.54/(1.6093*fuel_test)+R2*(abs(t1)-t1)+R2*(abs(t2)-t2)+R2*(abs(t3)-t3)+R2*(abs(grade_value)-grade_value) % unit is mpg  not L/100km
hc=hc_test/1.6093+R2*(abs(t1)-t1)+R2*(abs(t2)-t2)+R2*(abs(t3)-t3)+R2*(abs(grade_value)-grade_value)
co=co_test/1.6093+R2*(abs(t1)-t1)+R2*(abs(t2)-t2)+R2*(abs(t3)-t3)+R2*(abs(grade_value)-grade_value)
nox=nox_test/1.6093+R2*(abs(t1)-t1)+R2*(abs(t2)-t2)+R2*(abs(t3)-t3)+R2*(abs(grade_value)-grade_value)
 
 
fitness(i_pa,1)=0.7*fuel+0.1*hc+0.1*co+0.1*nox;
min_fuel(i_pa,1)=fuel;
min_hc(i_pa,1)=hc;
min_co(i_pa,1)=co;
min_nox(i_pa,1)=nox;
%///////////////////////////////////////
yyyy=yyyy+1;
             
         object(yyyy,1)=fuel;
         object(yyyy,2)=hc;
         object(yyyy,3)=co;
         object(yyyy,4)=nox;  
         
         
         
         performance(yyyy,1)=accel3;
         performance(yyyy,2)=accel1;
         performance(yyyy,3)=accel2;
         performance(yyyy,4)=grade_test;
         performance(yyyy,5)=delta_soc;
         
    
end
 
fitness_function(i_generation,1)=min(fitness);
fuel_function(i_generation,1)=min(min_fuel);
hc_function(i_generation,1)=min(min_hc);
co_function(i_generation,1)=min(min_co);
nox_function(i_generation,1)=min(min_nox);
%Evaluate objective funtion



%非占优排序
% Perform the non-dominated sorting procedure and caculate the 
%rank of each solution
 
 
%% function f = non_domination_sort_mod(x, M, V)
front_based=[];
for i_aa=1:2*N_size
    j_aa=1
    front_based(i_aa,j_aa)=0;
end
 
%
for i = 1 : 2*N_size
    % Number of individuals that dominate this individual
    individual(i).n = 0;
    % Individuals which this individual dominate
    individual(i).p = [];
    for j = 1 : 2*N_size
        dom_less = 0;
        dom_equal = 0;
        dom_more = 0;
        for k = 1 : M_object
            if (object(i,k) < object(j,k))
                dom_less = dom_less + 1;
            elseif (object(i,k) == object(j,k))
                dom_equal = dom_equal + 1;
            else
                dom_more = dom_more + 1;
            end
        end
        if dom_less == 0 & dom_equal ~= M_object
            individual(i).n = individual(i).n + 1;
        elseif dom_more == 0 & dom_equal ~= M_object
            individual(i).p = [individual(i).p j];
        end
    end   
    if individual(i).n == 0
         front_based(i,1) = 1;
        F(front).f = [F(front).f i];
    end
end
% Find the subsequent fronts
while ~isempty(F(front).f)
   Q = [];
   for i = 1 : length(F(front).f)
       if ~isempty(individual(F(front).f(i)).p)
            for j = 1 : length(individual(F(front).f(i)).p)
                individual(individual(F(front).f(i)).p(j)).n = ...
                    individual(individual(F(front).f(i)).p(j)).n - 1;
                if individual(individual(F(front).f(i)).p(j)).n == 0
                        front_based(individual(F(front).f(i)).p(j),1) = ...
                            front + 1;
                    Q = [Q individual(F(front).f(i)).p(j)];
                end
            end
       end
   end
   front =  front + 1;
   F(front).f = Q;
end
 
 
 
% b_sort_2N=[];% Sort the  pt and qt based on non-dominate
 
% for i_2N=1:front
%  %   clear 
%     b_sort_2N=[b_sort_2N F(i_2N).f];
% end
% 
% parent_population=[]
% for i_parent=1:length(b_sort_2N)
%     parent_population(i_parent,:)=R_conbine(b_sort_2N(i_parent),:)
%     
% end
[sorted_chromosome b_sort_2N]=sort(front_based)
 
 
object_based_front=[];
for i_object=1:1:length(b_sort_2N)
    object_based_front(i_object,:)=object(b_sort_2N(i_object),:)    
end
 
% for i_r2=1:N_size
%      results5(i_r2,:)=object(b_sort_2N(i_r2),:)
% end
% % % end
%Resize Rt from the size of 2N to N
 
% % % % [sorted_chromosome b_sort_2N]=sort(front_based)  %%%%%  delete
 
 
%////////////////////////////  
% caculation the crowed distance
 
%% Crowding distance
% Find the crowding distance for each individual in each front
current_index = 0;
 
 
crowed_distance=[];
% % % % % %   for III=1:2*N_size
% % % % % %      for JJJ=1    
% % % % % %        object(III,JJJ)=0;         
% % % % % %      end
% % % % % %   end
 
   distance_number=0;
 
 
 for front = 1 : (length(F) - 1)  % Tt is right 
    
               objective_crowed = [];
%                distance = 0;
              y = [];  
            previous_index = current_index + 1;
            for i_length = 1 : length(F(front).f)
           y(i_length,:) = object_based_front(current_index + i_length,:);
            end
          current_index = current_index + i_length;
        % Sort each individual based on the objective
           sorted_based_on_objective = [];
           index_of_objectives=[]; %%%%%%%%%%%%%%%%???  add
           
           for i = 1 : M_object
            [sorted_based_on_objective, index_of_objectives] = ...                   %%%%%%%%%%%%%%%%???  Index exceeds matrix dimensions
               sort(y(:,i));
              sorted_based_on_objective = [];
            for j = 1 : length(index_of_objectives)
              sorted_based_on_objective(j,:) = y(index_of_objectives(j),:);
            end
            f_max = ...
              sorted_based_on_objective(length(index_of_objectives),i);
              f_min = sorted_based_on_objective(1,i);
             y(index_of_objectives(length(index_of_objectives)),i)= Inf;
             y(index_of_objectives(1),i) = Inf;
                for j = 2 : length(index_of_objectives) - 1
                   next_obj  = sorted_based_on_objective(j + 1,i);
                   previous_obj  = sorted_based_on_objective(j - 1,i);
                 if (f_max - f_min == 0)
                   y(index_of_objectives(j), i) = Inf;
                
                 else
                    y(index_of_objectives(j), i) = ...
                       (next_obj - previous_obj)/(f_max - f_min);
                 end
                end
           end 
           distance = [];
           distance(:,1) = zeros(length(F(front).f),1);
           for i = 1 : M_object
             distance(:,1) = distance(:,1) + y(:,i);
           end
           
           for jj=1:length(F(front).f)
               crowed_distance(jj+distance_number,1)=distance(jj,1)
           end
           distance_number=distance_number+length(F(front).f)            
         
 end
     
     
 

%     y(:,M + V + 2) = distance;
%     y = y(:,1 : M + V + 2);
%     z(previous_index:current_index,:) = y;
% end
% f = z();
 
 
% /////////////////////  selection   选择
F_resize=[];
 
  for i_aaa=1:N_size
     for j_aaa=1    
       F_resize(i_aaa,j_aaa)=0;         
     end
  end
 
 
 previous_index = 0;
  for i = 1 : max_rank
      current_index = max(find(sorted_chromosome(:,1) == i));
      if current_index > N_size
          remaining = N_size - previous_index;
          temp_pop = ...
              b_sort_2N(previous_index + 1 : current_index, 1);
%           [temp_sort,temp_sort_index] = sort(crowed_distance(previous_index + 1:current_index, 1),'descend');   %  not support descend
            [temp_sort_f,temp_sort_index_f] = sort(crowed_distance(previous_index + 1:current_index, 1));
            
            [MM NN]=size(temp_sort_f);
            clear NN;
            for i_MM=1:MM
                temp_sort(i_MM,:)=temp_sort_f(MM+1-i_MM,:)
            end 
                
            [MMMM NNN]=size(temp_sort_index_f);
            clear NNN;
            for i_MMM=1:MMMM
                temp_sort_index(i_MMM,:)=temp_sort_index_f(MMMM+1-i_MMM,:)
            end 
                
         
         for j = 1 : remaining
             F_resize(previous_index + j,1) = temp_pop(temp_sort_index(j),1);
         end
         break;
%          return;
     elseif current_index <N_size
        F_resize(previous_index + 1 : current_index, 1) = ...
             b_sort_2N(previous_index + 1 : current_index, 1);
     else
          F_resize(previous_index + 1 : current_index, :) = ...
             b_sort_2N(previous_index + 1 : current_index, 1);
%         return;
     end
     previous_index = current_index;
  end
 
   
  for i=1:previous_index
      distance_slection(i,1)=crowed_distance(i,1)
  end 
  
  for i=1:N_size-previous_index
      distance_slection(i+previous_index,1)=temp_sort(i,1)
  end 
  
  
  
  %% Tournament selection process
 tour_size=2;
% Until the mating pool is filled, perform tournament selection
 
f_new_population=[];
for i = 1 :N_size
    % Select n individuals at random, where n = tour_size
    for j = 1 : tour_size
        % Select an individual at random
        candidate(j) = round(N_size*rand(1));
        % Make sure that the array starts from one. 
        if candidate(j) == 0
            candidate(j) = 1;
        end
        if j > 1
            % Make sure that same candidate is not choosen.
            while ~isempty(find(candidate(1 : j - 1) == candidate(j)))
                candidate(j) = round(N_size*rand(1));
                if candidate(j) == 0
                    candidate(j) = 1;
                end
            end
        end
    end
    % Collect information about the selected candidates.
    for j = 1 : tour_size
        c_obj_rank(j) = sorted_chromosome(candidate(j),1);
        c_obj_distance(j) = distance_slection(candidate(j),1);
    end
    % Find the candidate with the least rank
    min_candidate = ...
        find(c_obj_rank == min(c_obj_rank));
    % If more than one candiate have the least rank then find the candidate
    % within that group having the maximum crowding distance.
    if length(min_candidate) ~= 1
        max_candidate = ...
        find(c_obj_distance(min_candidate) == max(c_obj_distance(min_candidate)));
        % If a few individuals have the least rank and have maximum crowding
        % distance, select only one individual (not at random). 
        if length(max_candidate) ~= 1
            max_candidate = max_candidate(1);
        end
        % Add the selected individual to the mating pool
        f_new_population(i,1) = F_resize(candidate(min_candidate(max_candidate)),:);
    else
        % Add the selected individual to the mating pool
        f_new_population(i,1) = F_resize(candidate(min_candidate(1)),:);
    end
end
 
 
     parent_chromosome=[];  %%
 
for i_parent=1:length(F_resize)
    parent_chromosome(i_parent,:)=R_conbine(f_new_population(i_parent),:)
    
end
 
 
 
clear F;
 
 
end  % generation  
 
 
for i_r1=1:N_size
     results1(i_r1,:)=R_conbine(F_resize(i_r1),:);
 end
 
 
for i_r2=1:N_size
    results2(i_r2,:)=object(F_resize(i_r2),:)
end
 
 
for i_r3=1:N_size
    performance_out(i_r3,:)=performance(F_resize(i_r3),:)
end
 
 
  result=[results1 results2]
 
 resut_last=[results1 results2 performance_out]
 
 
 
 
figure(1);
plot(results2(:,2),results2(:,1),'o')
xlabel('hc grams/km');
ylabel('fuel L/100km');
 
figure(2);
plot(results2(:,3),results2(:,1),'o')
xlabel('co grams/km');
ylabel('fuel L/100km');
 
figure(3);
plot(results2(:,4),results2(:,1),'o')
xlabel('nox grams/km');
ylabel('fuel L/100km');
 
figure(4);
plot(results2(:,1),results2(:,2),'o')
xlabel('fuel L/100km');
ylabel('hc grams/km');
 
figure(5);
plot(results2(:,3),results2(:,2),'o')
xlabel('co grams/km');
ylabel('hc grams/km');
 
figure(6);
plot(results2(:,4),results2(:,2),'o')
xlabel('nox grams/km');
ylabel('hc grams/km');
 
 
figure(7);
plot(results2(:,1),results2(:,3),'o')
xlabel('fuel L/100km');
ylabel('co grams/km');
 
figure(8);
plot(results2(:,2),results2(:,3),'o')
xlabel('hc grams/km');
ylabel('co grams/km');
 
figure(9);
plot(results2(:,4),results2(:,3),'o')
xlabel('nox grams/km');
ylabel('co grams/km');
 
figure(10);
plot(results2(:,1),results2(:,4),'o')
xlabel('fuel L/100km');
ylabel('nox grams/km');
 
figure(11);
plot(results2(:,2),results2(:,4),'o')
xlabel('hc grams/km');
ylabel('nox grams/km');
 
figure(12);
plot(results2(:,3),results2(:,4),'o')
xlabel('co grams/km');
ylabel('nox grams/km');
 
 
 
%fitness funtion
www=[];
for i_shu=1:G_number
    for j_shu=1:1
     www(i_shu,j_shu)=i_shu;
   end
end
 
 
zzz=[];
for i_zhi=1:N_size
    for j_zhi=1:1
        zzz(i_zhi,j_zhi)=i_zhi;
    end
end
 
 
figure(13);
plot(zzz(:,1),results2(:,1),'o')
xlabel('solution');
ylabel('fuel L/100km');
 
 
figure(14);
plot(zzz(:,1),results2(:,2),'o')
xlabel('solution');
ylabel('hc grams/km');
 
figure(15);
plot(zzz(:,1),results2(:,3),'o')
xlabel('solution');
ylabel('co grams/km');
 
figure(16);
plot(zzz(:,1),results2(:,4),'o')
xlabel('solution');
ylabel('nox grams/km');
 
 
figure(17);
plot(www(:,1),fitness_function(:,1),'o')
xlabel('generation');
ylabel('fitness');
 
for i_p=1:N_size
    performance_based(i_p,:)=performance(F_resize(i_p),:)
end
 