% Method for nearest neighbor classifier
function [retClasses] = classifyNN(manip_test, manip_train, gnd_Train)
    retClasses = zeros(1,size(manip_test,2));
    calcedDist = calcDistanceNN(manip_test, manip_train);
    
    % Supplied test example length
    for i=1:size(manip_test,2)
        
       % nearest neighborsby ascending sorting
       [d, ind]= sort(calcedDist(i,:),'ascend');
       small_dist = d(1:2);
       k_nearest = gnd_Train(ind(1:2));
       vote_classes = zeros(size(k_nearest));
       
       % count votes for each class
       for j=1:size(k_nearest')
            class = k_nearest(j);
            if ismember(class, find(vote_classes))
                vote_classes(class) = vote_classes(class) + 1;
            else
                vote_classes(class) = 1;
            end
       end
       
       % majority voted class
       nz_classes = vote_classes(vote_classes~=0);
       if all(nz_classes==nz_classes(1))
          majority = gnd_Train(ind(1:1));
       else 
           [val, vote_lb] = sort(vote_classes,'descend');
           
           big_vote_lb = vote_lb(val==val(1:1));
           [~,loc] = ismember(big_vote_lb,k_nearest);
           [~,d_loc] = min(small_dist(loc)); % min dist
           majority = big_vote_lb(d_loc);
       end
       retClasses(i)= majority;
    end
end

% ED between two generated sets NN version
function ret = calcDistanceNN(manip_test, manip_train)
    calcDist = zeros(size(manip_test,2),size(manip_train,2));
    for i=1:size(manip_test,2)
        for j=1:size(manip_train,2)
            calcDist(i,j) = norm(manip_test(:,i) - manip_train(:, j));
        end
    end
    ret = calcDist
end

