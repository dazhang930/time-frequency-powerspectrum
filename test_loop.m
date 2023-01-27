
logicalArray = true(trials,1);
mMall = [];
% z = zeros(channel, 2*n)
for k = 1:channel
%     my_field = strcat('v',num2str(k));

    assignin('base',['M',num2str(k)],squeeze(d(k,:,326:375)))
%     all(  isnan(mean( squeeze(d(k,:,326:375)),1 ))) 
%     all( isnan(mean( squeeze(d(k,:,326:375)),2 )) )
    assignin('base',['mM',num2str(k)], ~isnan(mean( squeeze(d(k,:,326:375)),2 )))
    each_logic = (evalin('base',['mM',num2str(k)]));
    logicalArray = logicalArray & each_logic;

end

logicalArray = true(trials,1);
for k = 1:channel
%     my_field = strcat('v',num2str(k));

    each_M = (evalin('base',['M',num2str(k)]));
    assignin('base',['valueM',num2str(k)], each_M(find(logicalArray),:))
    
    mean_value_M = evalin('base',['valueM',num2str(k)]);
    temp = mean(mean_value_M, 2)';
    size(temp)
    mMall = [mMall, temp'];

end

mMall = mMall'
count = 0;
newmMall = []
logicalTrial = true(trials,1);
for t = 1:trials
    if all( isnan(mMall(:, t) )) 
%         disp("good")
        count = count+1;
        logicalTrial(t) = false;
    else
        newmMall = [newmMall, mMall(:, t)];
    end
end

disp("newmMall" )
disp(size(newmMall))

empty_channel=0;
finalMall = []
for c = 1:channel
    if any(isnan(newmMall(c, :) ))
%         disp("good")
        empty_channel = empty_channel+1
    else
        finalMall = [finalMall, newmMall(c, :)'];
    end
end

disp("final channel")
disp(empty_channel)
disp(size(finalMall))
finalMall = finalMall';


figure;imagesc(mMall)
figure;imagesc(newmMall)
figure;imagesc(finalMall)
save('~/Downloads/channel52_mean.mat','mMall','tr','-v7.3')
save('~/Downloads/channel37_mean.mat','finalMall','-v7.3')


target_animal = tr.ctg_animal(find(logicalTrial),:);
target_cloth = tr.ctg_cloth(find(logicalTrial),:);
target_device = tr.ctg_device(find(logicalTrial),:);
target_food = tr.ctg_food(find(logicalTrial),:);
target_holiday = tr.ctg_holiday(find(logicalTrial),:);
target_insect = tr.ctg_insect(find(logicalTrial),:);
target_instrument = tr.ctg_instrument(find(logicalTrial),:);
target_organ = tr.ctg_organ(find(logicalTrial),:);
target_person = tr.ctg_person(find(logicalTrial),:);
target_place = tr.ctg_place(find(logicalTrial),:);
target_thing = tr.ctg_thing(find(logicalTrial),:);
target_tool = tr.ctg_tool(find(logicalTrial),:);
target_weapon = tr.ctg_weapon(find(logicalTrial),:);
target_animate = tr.ctg_animate(find(logicalTrial),:);
target_inanimate = tr.ctg_inanimate(find(logicalTrial),:);
target_thing = tr.ctg_thing(find(logicalTrial),:);

mMall_target = cat(2, target_animal, target_cloth, target_device, target_food, target_holiday, target_insect, target_instrument, target_organ, target_person, target_place,target_thing, target_tool, target_weapon, target_animate, target_inanimate );
size(mMall_target)
save('~/Downloads/channel37_target.mat','mMall_target','-v7.3')