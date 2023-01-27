[d2,q2,tr2,ch2,frx2]=AN_mobilizedata2('EC137','AN',{'hp','mtg'});


% M1 = squeeze(d(1,:,326:375));
% M2 = squeeze(d(2,:,326:375));
% M3 = squeeze(d(3,:,326:375));
% M4 = squeeze(d(4,:,326:375));
% M5 = squeeze(d(5,:,326:375));
% M6 = squeeze(d(6,:,326:375));
% M7 = squeeze(d(7,:,326:375));
% M8 = squeeze(d(8,:,326:375));
% M9 = squeeze(d(9,:,326:375));
% 
% mM1 = ~isnan(mean(M1,2))
% mM1 = ~isnan(mean(M1,2));
% mM2 = ~isnan(mean(M2,2));
% mM3 = ~isnan(mean(M3,2));
% mM4 = ~isnan(mean(M4,2));
% mM5 = ~isnan(mean(M5,2));
% mM6 = ~isnan(mean(M6,2));
% mM7 = ~isnan(mean(M7,2));
% mM8 = ~isnan(mean(M8,2));
% mM9 = ~isnan(mean(M9,2));
% mMAnd = mM1&mM2&mM3&mM4&mM5&mM6&mM7& mM8 & mM9;
% valueM1 = M1(find(mMAnd),:);
% valueM2 = M2(find(mMAnd),:);
% valueM3 = M3(find(mMAnd),:);
% valueM4 = M4(find(mMAnd),:);
% valueM5 = M5(find(mMAnd),:);
% valueM6 = M6(find(mMAnd),:);
% valueM7 = M7(find(mMAnd),:);
% valueM8 = M8(find(mMAnd),:);
% valueM9 = M9(find(mMAnd),:);
% mM1 = mean(valueM1, 2)';
% mM2 = mean(valueM2, 2)';
% mM3 = mean(valueM3, 2)';
% mM4 = mean(valueM4, 2)';
% mM5 = mean(valueM5, 2)';
% mM6 = mean(valueM6, 2)';
% mM7 = mean(valueM7, 2)';
% mM8 = mean(valueM8, 2)';
% mM9 = mean(valueM9, 2)';
% mMall = cat(1, mM1, mM2, mM3, mM4, mM5, mM6, mM7, mM8, mM9);
% figure;imagesc(mMall)
% save('~/Downloads/channel_mean.mat','mMall','-v7.3')
% target_animal = tr.ctg_animal(find(mMAnd),:);
% size(target_animal)
% target_cloth = tr.ctg_cloth(find(mMAnd),:);
% target_device = tr.ctg_device(find(mMAnd),:);
% target_food = tr.ctg_food(find(mMAnd),:);
% target_holiday = tr.ctg_holiday(find(mMAnd),:);
% target_insect = tr.ctg_insect(find(mMAnd),:);
% target_instrument = tr.ctg_instrument(find(mMAnd),:);
% target_organ = tr.ctg_organ(find(mMAnd),:);
% target_person = tr.ctg_person(find(mMAnd),:);
% target_place = tr.ctg_place(find(mMAnd),:);
% target_thing = tr.ctg_thing(find(mMAnd),:);
% target_tool = tr.ctg_tool(find(mMAnd),:);
% target_weapon = tr.ctg_weapon(find(mMAnd),:);
% target_animate = tr.ctg_animate(find(mMAnd),:);
% target_inanimate = tr.ctg_inanimate(find(mMAnd),:);
% target_thing = tr.ctg_thing(find(mMAnd),:);
% mMall_target = cat(2, target_animal, target_cloth, target_device, target_food, target_holiday, target_insect, target_instrument, target_organ, target_person, target_place,target_thing, target_tool, target_weapon, target_animate, target_inanimate );
% size(mMall_target)
% save('~/Downloads/channel_target.mat','mMall_target','-v7.3')