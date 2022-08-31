clear all 
clc
% clea all 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filehead = "/media/faraday/andong/SW_synthetic/SW_Storm_";
filehead_save = "/media/faraday/andong/SW_synthetic/preprocess/SW_Storm_";
fileend = "_sb_10.mat";
% filename = 'Data/SW_Storm_2_sb_10.mat'

for n = 1:2
    filename = filehead+n+fileend;
    savename = filehead_save+n+fileend;
    try
        data = load(filename);
    catch
        continue;
    end

    year = data.last_orig_time.Year;
    month = data.last_orig_time.Month;
    day = data.last_orig_time.Day;
    hours = data.last_orig_time.Hour;
    minutes = data.last_orig_time.Minute;

    year_clu = zeros(size(data.orig.Time, 1), 1);
    month_clu = zeros(size(data.orig.Time, 1), 1);
    day_clu = zeros(size(data.orig.Time, 1), 1);
    doy_clu = zeros(size(data.orig.Time, 1), 1);
    hour_clu = zeros(size(data.orig.Time, 1), 1);
    minute_clu = zeros(size(data.orig.Time, 1), 1);

    for i = 1:size(data.orig.Time, 1)
        year_clu(i) = data.orig.Time(i).Year;
        month_clu(i) = data.orig.Time(i).Month;
        day_clu(i) = data.orig.Time(i).Day;
        hour_clu(i) = data.orig.Time(i).Hour;
        minute_clu(i) = data.orig.Time(i).Minute;

        doy_clu(i) = datenum([year_clu(i), month_clu(i), day_clu(i)]) - datenum(year_clu(i),1,1) + 1;

    end

    Bx = data.orig.Bx;
    By = data.orig.By;
    Bz = data.orig.Bz;
    N = data.orig.density;
    V = data.orig.V;
    B_norm = sqrt(Bx.^2+By.^2);
    Dst = circshift(data.orig.Dst, -1);
    UTC = hour_clu+minute_clu/60;

    t_year = 23.4*cos((doy_clu-172)*2*pi/365.25);
    t_day = 11.2*cos((UTC-16.72)*2*pi/24);
    t = t_year+t_day;
    theta_c = atan2(By, Bz);


    data.vari = [N, V, B_norm, Bz, t, sin(theta_c), Dst];

    data.date = [year_clu month_clu day_clu hour_clu minute_clu];
    data.last_orig_time = [year, month, day, hours, minutes];

    save(savename, '-struct', 'data');
end
